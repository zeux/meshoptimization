// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct BasisSettings
{
	int etc1s_l;
	int etc1s_q;
	int uastc_l;
	float uastc_q;
};

static const char* kMimeTypes[][2] = {
    {"image/jpeg", ".jpg"},
    {"image/jpeg", ".jpeg"},
    {"image/png", ".png"},
};

static const BasisSettings kBasisSettings[10] = {
    {1, 1, 0, 1.5f},
    {1, 6, 0, 1.f},
    {1, 20, 1, 1.0f},
    {1, 50, 1, 0.75f},
    {1, 90, 1, 0.5f},
    {1, 128, 1, 0.4f},
    {1, 160, 1, 0.34f},
    {1, 192, 1, 0.29f}, // default
    {1, 224, 2, 0.26f},
    {1, 255, 2, 0.f},
};

static const char* inferMimeType(const char* path)
{
	std::string ext = getExtension(path);

	for (size_t i = 0; i < sizeof(kMimeTypes) / sizeof(kMimeTypes[0]); ++i)
		if (ext == kMimeTypes[i][1])
			return kMimeTypes[i][0];

	return "";
}

static bool parseDataUri(const char* uri, std::string& mime_type, std::string& result)
{
	if (strncmp(uri, "data:", 5) == 0)
	{
		const char* comma = strchr(uri, ',');

		if (comma && comma - uri >= 7 && strncmp(comma - 7, ";base64", 7) == 0)
		{
			const char* base64 = comma + 1;
			size_t base64_size = strlen(base64);
			size_t size = base64_size - base64_size / 4;

			if (base64_size >= 2)
			{
				size -= base64[base64_size - 2] == '=';
				size -= base64[base64_size - 1] == '=';
			}

			void* data = 0;

			cgltf_options options = {};
			cgltf_result res = cgltf_load_buffer_base64(&options, size, base64, &data);

			if (res != cgltf_result_success)
				return false;

			mime_type = std::string(uri + 5, comma - 7);
			result = std::string(static_cast<const char*>(data), size);

			free(data);

			return true;
		}
	}

	return false;
}

bool readImage(const cgltf_image& image, const char* input_path, std::string& data, std::string& mime_type)
{
	if (image.uri && parseDataUri(image.uri, mime_type, data))
	{
		return true;
	}
	else if (image.buffer_view && image.buffer_view->buffer->data && image.mime_type)
	{
		const cgltf_buffer_view* view = image.buffer_view;

		data.assign(static_cast<const char*>(view->buffer->data) + view->offset, view->size);
		mime_type = image.mime_type;
		return true;
	}
	else if (image.uri && *image.uri)
	{
		std::string path = image.uri;

		cgltf_decode_uri(&path[0]);
		path.resize(strlen(&path[0]));

		mime_type = image.mime_type ? image.mime_type : inferMimeType(path.c_str());

		return readFile(getFullPath(path.c_str(), input_path).c_str(), data);
	}
	else
	{
		return false;
	}
}

static int readInt16(const std::string& data, size_t offset)
{
	return (unsigned char)data[offset] * 256 + (unsigned char)data[offset + 1];
}

static int readInt32(const std::string& data, size_t offset)
{
	return (unsigned((unsigned char)data[offset]) << 24) |
	       (unsigned((unsigned char)data[offset + 1]) << 16) |
	       (unsigned((unsigned char)data[offset + 2]) << 8) |
	       unsigned((unsigned char)data[offset + 3]);
}

static bool getDimensionsPng(const std::string& data, int& width, int& height)
{
	if (data.size() < 8 + 8 + 13 + 4)
		return false;

	const char* signature = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a";
	if (data.compare(0, 8, signature) != 0)
		return false;

	if (data.compare(12, 4, "IHDR") != 0)
		return false;

	width = readInt32(data, 16);
	height = readInt32(data, 20);

	return true;
}

static bool getDimensionsJpeg(const std::string& data, int& width, int& height)
{
	size_t offset = 0;

	// note, this can stop parsing before reaching the end but we stop at SOF anyway
	while (offset + 4 <= data.size())
	{
		if (data[offset] != '\xff')
			return false;

		char marker = data[offset + 1];

		if (marker == '\xff')
		{
			offset++;
			continue; // padding
		}

		// d0..d9 correspond to SOI, RSTn, EOI
		if (marker == 0 || unsigned(marker - '\xd0') <= 9)
		{
			offset += 2;
			continue; // no payload
		}

		// c0..c1 correspond to SOF0, SOF1
		if (marker == '\xc0' || marker == '\xc2')
		{
			if (offset + 10 > data.size())
				return false;

			width = readInt16(data, offset + 7);
			height = readInt16(data, offset + 5);

			return true;
		}

		offset += 2 + readInt16(data, offset + 2);
	}

	return false;
}

static bool hasTransparencyPng(const std::string& data)
{
	if (data.size() < 8 + 8 + 13 + 4)
		return false;

	const char* signature = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a";
	if (data.compare(0, 8, signature) != 0)
		return false;

	if (data.compare(12, 4, "IHDR") != 0)
		return false;

	int ctype = data[25];

	if (ctype != 3)
		return ctype == 4 || ctype == 6;

	size_t offset = 8; // reparse IHDR chunk for simplicity

	while (offset + 12 <= data.size())
	{
		int length = readInt32(data, offset);

		if (length < 0)
			return false;

		if (data.compare(offset + 4, 4, "tRNS") == 0)
			return true;

		offset += 12 + length;
	}

	return false;
}

bool hasAlpha(const std::string& data, const char* mime_type)
{
	if (strcmp(mime_type, "image/png") == 0)
		return hasTransparencyPng(data);
	else
		return false;
}

static const char* mimeExtension(const char* mime_type)
{
	for (size_t i = 0; i < sizeof(kMimeTypes) / sizeof(kMimeTypes[0]); ++i)
		if (strcmp(kMimeTypes[i][0], mime_type) == 0)
			return kMimeTypes[i][1];

	return ".raw";
}

static bool getDimensions(const std::string& data, const char* mime_type, int& width, int& height)
{
	if (strcmp(mime_type, "image/png") == 0)
		return getDimensionsPng(data, width, height);
	if (strcmp(mime_type, "image/jpeg") == 0)
		return getDimensionsJpeg(data, width, height);

	return false;
}

static int roundPow2(int value)
{
	int result = 1;

	while (result < value)
		result <<= 1;

	// to prevent odd texture sizes from increasing the size too much, we round to nearest power of 2 above a certain size
	if (value > 128 && result * 3 / 4 > value)
		result >>= 1;

	return result;
}

static int roundBlock(int value, bool pow2)
{
	if (value == 0)
		return 4;

	if (pow2 && value > 4)
		return roundPow2(value);

	return (value + 3) & ~3;
}

#ifdef __wasi__
static int execute(const char* cmd, bool ignore_stdout, bool ignore_stderr)
{
	return system(cmd);
}

static std::string getExecutable(const char* name, const char* env)
{
	return name;
}
#else
static int execute(const char* cmd_, bool ignore_stdout, bool ignore_stderr)
{
#ifdef _WIN32
	std::string ignore = "nul";
#else
	std::string ignore = "/dev/null";
#endif

	std::string cmd = cmd_;

	if (ignore_stdout)
		(cmd += " >") += ignore;
	if (ignore_stderr)
		(cmd += " 2>") += ignore;

	return system(cmd.c_str());
}

static std::string getExecutable(const char* name, const char* env)
{
	const char* path = getenv(env);
	path = path ? path : name;

#ifdef _WIN32
	// when the executable path contains a space, we need to quote the path ourselves
	if (path[0] != '"' && strchr(path, ' '))
		return '"' + std::string(path) + '"';
#endif

	return path;
}
#endif

bool checkBasis(bool verbose)
{
#ifdef WITH_BASISU
	(void)verbose;
	return true;
#else
	std::string cmd = getExecutable("basisu", "BASISU_PATH");

	cmd += " -version";

	int rc = execute(cmd.c_str(), /* ignore_stdout= */ !verbose, /* ignore_stderr= */ !verbose);
	if (verbose)
		printf("%s => %d\n", cmd.c_str(), rc);

	return rc == 0;
#endif
}

bool encodeBasisInternal(const char* input, const char* output, bool yflip, bool normal_map, bool linear, bool uastc, int uastc_l, float uastc_q, int etc1s_l, int etc1s_q, int zstd_l, int width, int height);

bool encodeBasis(const std::string& data, const char* mime_type, std::string& result, const ImageInfo& info, const Settings& settings)
{
	// TODO: Support texture_scale and texture_pow2 via new -resample switch from https://github.com/BinomialLLC/basis_universal/pull/226
	TempFile temp_input(mimeExtension(mime_type));
	TempFile temp_output(".ktx2");

	if (!writeFile(temp_input.path.c_str(), data))
		return false;

	int quality = settings.texture_quality[info.kind];
	bool uastc = settings.texture_uastc[info.kind];

	const BasisSettings& bs = kBasisSettings[quality - 1];

#ifdef WITH_BASISU
	int width = 0, height = 0;
	if (!getDimensions(data, mime_type, width, height))
		return false;

	int newWidth = roundBlock(int(width * settings.texture_scale), settings.texture_pow2);
	int newHeight = roundBlock(int(height * settings.texture_scale), settings.texture_pow2);

	int zstd = uastc ? 9 : 0;

	bool ok = encodeBasisInternal(temp_input.path.c_str(), temp_output.path.c_str(), settings.texture_flipy, info.normal_map, !info.srgb, uastc, bs.uastc_l, bs.uastc_q, bs.etc1s_l, bs.etc1s_q, zstd, newWidth, newHeight);

	return ok && readFile(temp_output.path.c_str(), result);
#else
	std::string cmd = getExecutable("basisu", "BASISU_PATH");

	cmd += " -mipmap";

	if (settings.texture_flipy)
		cmd += " -y_flip";

	if (info.normal_map)
	{
		cmd += " -normal_map";
		// for optimal quality we should specify seperate_rg_to_color_alpha but this requires renderer awareness
	}
	else if (!info.srgb)
	{
		cmd += " -linear";
	}

	if (uastc)
	{
		char cs[128];
		sprintf(cs, " -uastc_level %d -uastc_rdo_l %.2f", bs.uastc_l, bs.uastc_q);

		cmd += " -uastc";
		cmd += cs;
		cmd += " -uastc_rdo_d 1024";
	}
	else
	{
		char cs[128];
		sprintf(cs, " -comp_level %d -q %d", bs.etc1s_l, bs.etc1s_q);

		cmd += cs;
	}

	cmd += " -ktx2";

	if (uastc)
		cmd += " -ktx2_zstandard_level 9";

	cmd += " -file ";
	cmd += temp_input.path;
	cmd += " -output_file ";
	cmd += temp_output.path;

	int rc = execute(cmd.c_str(), /* ignore_stdout= */ true, /* ignore_stderr= */ false);
	if (settings.verbose > 1)
		printf("%s => %d\n", cmd.c_str(), rc);

	return rc == 0 && readFile(temp_output.path.c_str(), result);
#endif
}

bool checkKtx(bool verbose)
{
#ifdef WITH_BASISU
	(void)verbose;
	return false;
#else
	std::string cmd = getExecutable("toktx", "TOKTX_PATH");

	cmd += " --version";

	int rc = execute(cmd.c_str(), /* ignore_stdout= */ !verbose, /* ignore_stderr= */ !verbose);
	if (verbose)
		printf("%s => %d\n", cmd.c_str(), rc);

	return rc == 0;
#endif
}

bool encodeKtx(const std::string& data, const char* mime_type, std::string& result, const ImageInfo& info, const Settings& settings)
{
	int width = 0, height = 0;
	if (!getDimensions(data, mime_type, width, height))
		return false;

	TempFile temp_input(mimeExtension(mime_type));
	TempFile temp_output(".ktx2");

	if (!writeFile(temp_input.path.c_str(), data))
		return false;

	std::string cmd = getExecutable("toktx", "TOKTX_PATH");

	int quality = settings.texture_quality[info.kind];
	bool uastc = settings.texture_uastc[info.kind];

	const BasisSettings& bs = kBasisSettings[quality - 1];

	cmd += " --t2";
	cmd += " --2d";
	cmd += " --genmipmap";
	cmd += " --nowarn";

	int newWidth = roundBlock(int(width * settings.texture_scale), settings.texture_pow2);
	int newHeight = roundBlock(int(height * settings.texture_scale), settings.texture_pow2);

	if (newWidth != width || newHeight != height)
	{
		char wh[128];
		sprintf(wh, " --resize %dx%d", newWidth, newHeight);
		cmd += wh;
	}

	if (settings.texture_flipy)
		cmd += " --lower_left_maps_to_s0t0";

	if (uastc)
	{
		char cs[128];
		sprintf(cs, " %d --uastc_rdo_l %.2f", bs.uastc_l, bs.uastc_q);

		cmd += " --uastc";
		cmd += cs;
		cmd += " --uastc_rdo_d 1024";
		cmd += " --zcmp 9";
	}
	else
	{
		char cs[128];
		sprintf(cs, " --clevel %d --qlevel %d", bs.etc1s_l, bs.etc1s_q);

		cmd += " --bcmp";
		cmd += cs;

		// for optimal quality we should specify separate_rg_to_color_alpha but this requires renderer awareness
		if (info.normal_map)
			cmd += " --normal_map";
	}

	if (info.srgb)
		cmd += " --srgb";
	else
		cmd += " --linear";

	cmd += " -- ";
	cmd += temp_output.path;
	cmd += " ";
	cmd += temp_input.path;

	int rc = execute(cmd.c_str(), /* ignore_stdout= */ false, /* ignore_stderr= */ false);
	if (settings.verbose > 1)
		printf("%s => %d\n", cmd.c_str(), rc);

	return rc == 0 && readFile(temp_output.path.c_str(), result);
}
