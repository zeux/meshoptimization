// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <algorithm>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "../src/meshoptimizer.h"

struct Bounds
{
	Attr min, max;

	Bounds()
	{
		min.f[0] = min.f[1] = min.f[2] = min.f[3] = +FLT_MAX;
		max.f[0] = max.f[1] = max.f[2] = max.f[3] = -FLT_MAX;
	}

	bool isValid() const
	{
		return min.f[0] <= max.f[0] && min.f[1] <= max.f[1] && min.f[2] <= max.f[2] && min.f[3] <= max.f[3];
	}
};

static void updateAttributeBounds(const Mesh& mesh, cgltf_attribute_type type, Bounds& b)
{
	Attr pad = {};

	for (size_t j = 0; j < mesh.streams.size(); ++j)
	{
		const Stream& s = mesh.streams[j];

		if (s.type == type)
		{
			if (s.target == 0)
			{
				for (size_t k = 0; k < s.data.size(); ++k)
				{
					const Attr& a = s.data[k];

					b.min.f[0] = std::min(b.min.f[0], a.f[0]);
					b.min.f[1] = std::min(b.min.f[1], a.f[1]);
					b.min.f[2] = std::min(b.min.f[2], a.f[2]);
					b.min.f[3] = std::min(b.min.f[3], a.f[3]);

					b.max.f[0] = std::max(b.max.f[0], a.f[0]);
					b.max.f[1] = std::max(b.max.f[1], a.f[1]);
					b.max.f[2] = std::max(b.max.f[2], a.f[2]);
					b.max.f[3] = std::max(b.max.f[3], a.f[3]);
				}
			}
			else
			{
				for (size_t k = 0; k < s.data.size(); ++k)
				{
					const Attr& a = s.data[k];

					pad.f[0] = std::max(pad.f[0], fabsf(a.f[0]));
					pad.f[1] = std::max(pad.f[1], fabsf(a.f[1]));
					pad.f[2] = std::max(pad.f[2], fabsf(a.f[2]));
					pad.f[3] = std::max(pad.f[3], fabsf(a.f[3]));
				}
			}
		}
	}

	for (int k = 0; k < 4; ++k)
	{
		b.min.f[k] -= pad.f[k];
		b.max.f[k] += pad.f[k];
	}
}

QuantizationPosition prepareQuantizationPosition(const std::vector<Mesh>& meshes, const Settings& settings)
{
	QuantizationPosition result = {};

	result.bits = settings.pos_bits;

	Bounds b;

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		updateAttributeBounds(meshes[i], cgltf_attribute_type_position, b);
	}

	if (b.isValid())
	{
		result.offset[0] = b.min.f[0];
		result.offset[1] = b.min.f[1];
		result.offset[2] = b.min.f[2];
		result.scale = std::max(b.max.f[0] - b.min.f[0], std::max(b.max.f[1] - b.min.f[1], b.max.f[2] - b.min.f[2]));
	}

	return result;
}

void prepareQuantizationTexture(cgltf_data* data, std::vector<QuantizationTexture>& result, const std::vector<Mesh>& meshes, const Settings& settings)
{
	std::vector<Bounds> bounds(result.size());

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		const Mesh& mesh = meshes[i];

		if (!mesh.material)
			continue;

		size_t mi = mesh.material - data->materials;
		assert(mi < bounds.size());

		updateAttributeBounds(mesh, cgltf_attribute_type_texcoord, bounds[mi]);
	}

	for (size_t i = 0; i < result.size(); ++i)
	{
		QuantizationTexture& qt = result[i];

		qt.bits = settings.tex_bits;

		const Bounds& b = bounds[i];

		if (b.isValid())
		{
			qt.offset[0] = b.min.f[0];
			qt.offset[1] = b.min.f[1];
			qt.scale[0] = b.max.f[0] - b.min.f[0];
			qt.scale[1] = b.max.f[1] - b.min.f[1];
		}
	}
}

void getPositionBounds(float min[3], float max[3], const Stream& stream, const QuantizationPosition* qp)
{
	assert(stream.type == cgltf_attribute_type_position);
	assert(stream.data.size() > 0);

	min[0] = min[1] = min[2] = FLT_MAX;
	max[0] = max[1] = max[2] = -FLT_MAX;

	for (size_t i = 0; i < stream.data.size(); ++i)
	{
		const Attr& a = stream.data[i];

		for (int k = 0; k < 3; ++k)
		{
			min[k] = std::min(min[k], a.f[k]);
			max[k] = std::max(max[k], a.f[k]);
		}
	}

	if (qp)
	{
		float pos_rscale = qp->scale == 0.f ? 0.f : 1.f / qp->scale;

		for (int k = 0; k < 3; ++k)
		{
			if (stream.target == 0)
			{
				min[k] = float(meshopt_quantizeUnorm((min[k] - qp->offset[k]) * pos_rscale, qp->bits));
				max[k] = float(meshopt_quantizeUnorm((max[k] - qp->offset[k]) * pos_rscale, qp->bits));
			}
			else
			{
				min[k] = (min[k] >= 0.f ? 1.f : -1.f) * float(meshopt_quantizeUnorm(fabsf(min[k]) * pos_rscale, qp->bits));
				max[k] = (max[k] >= 0.f ? 1.f : -1.f) * float(meshopt_quantizeUnorm(fabsf(max[k]) * pos_rscale, qp->bits));
			}
		}
	}
}

static void renormalizeWeights(uint8_t (&w)[4])
{
	int sum = w[0] + w[1] + w[2] + w[3];

	if (sum == 255)
		return;

	// we assume that the total error is limited to 0.5/component = 2
	// this means that it's acceptable to adjust the max. component to compensate for the error
	int max = 0;

	for (int k = 1; k < 4; ++k)
		if (w[k] > w[max])
			max = k;

	w[max] += uint8_t(255 - sum);
}

static void encodeOct(int& fu, int& fv, float nx, float ny, float nz, int bits)
{
	float nl = fabsf(nx) + fabsf(ny) + fabsf(nz);
	float ns = nl == 0.f ? 0.f : 1.f / nl;

	nx *= ns;
	ny *= ns;

	float u = (nz >= 0.f) ? nx : (1 - fabsf(ny)) * (nx >= 0.f ? 1.f : -1.f);
	float v = (nz >= 0.f) ? ny : (1 - fabsf(nx)) * (ny >= 0.f ? 1.f : -1.f);

	fu = meshopt_quantizeSnorm(u, bits);
	fv = meshopt_quantizeSnorm(v, bits);
}

static StreamFormat writeVertexStreamRaw(std::string& bin, const Stream& stream, cgltf_type type, size_t components)
{
	assert(components >= 1 && components <= 4);

	for (size_t i = 0; i < stream.data.size(); ++i)
	{
		const Attr& a = stream.data[i];

		bin.append(reinterpret_cast<const char*>(a.f), sizeof(float) * components);
	}

	StreamFormat format = {type, cgltf_component_type_r_32f, false, sizeof(float) * components};
	return format;
}

StreamFormat writeVertexStream(std::string& bin, const Stream& stream, const QuantizationPosition& qp, const QuantizationTexture& qt, const Settings& settings)
{
	if (stream.type == cgltf_attribute_type_position)
	{
		if (!settings.quantize)
			return writeVertexStreamRaw(bin, stream, cgltf_type_vec3, 3);

		if (stream.target == 0)
		{
			float pos_rscale = qp.scale == 0.f ? 0.f : 1.f / qp.scale;

			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				const Attr& a = stream.data[i];

				uint16_t v[4] = {
				    uint16_t(meshopt_quantizeUnorm((a.f[0] - qp.offset[0]) * pos_rscale, qp.bits)),
				    uint16_t(meshopt_quantizeUnorm((a.f[1] - qp.offset[1]) * pos_rscale, qp.bits)),
				    uint16_t(meshopt_quantizeUnorm((a.f[2] - qp.offset[2]) * pos_rscale, qp.bits)),
				    0};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}

			StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_16u, false, 8};
			return format;
		}
		else
		{
			float pos_rscale = qp.scale == 0.f ? 0.f : 1.f / qp.scale;

			int maxv = 0;

			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				const Attr& a = stream.data[i];

				maxv = std::max(maxv, meshopt_quantizeUnorm(fabsf(a.f[0]) * pos_rscale, qp.bits));
				maxv = std::max(maxv, meshopt_quantizeUnorm(fabsf(a.f[1]) * pos_rscale, qp.bits));
				maxv = std::max(maxv, meshopt_quantizeUnorm(fabsf(a.f[2]) * pos_rscale, qp.bits));
			}

			if (maxv <= 127)
			{
				for (size_t i = 0; i < stream.data.size(); ++i)
				{
					const Attr& a = stream.data[i];

					int8_t v[4] = {
					    int8_t((a.f[0] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[0]) * pos_rscale, qp.bits)),
					    int8_t((a.f[1] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[1]) * pos_rscale, qp.bits)),
					    int8_t((a.f[2] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[2]) * pos_rscale, qp.bits)),
					    0};
					bin.append(reinterpret_cast<const char*>(v), sizeof(v));
				}

				StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_8, false, 4};
				return format;
			}
			else
			{
				for (size_t i = 0; i < stream.data.size(); ++i)
				{
					const Attr& a = stream.data[i];

					int16_t v[4] = {
					    int16_t((a.f[0] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[0]) * pos_rscale, qp.bits)),
					    int16_t((a.f[1] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[1]) * pos_rscale, qp.bits)),
					    int16_t((a.f[2] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[2]) * pos_rscale, qp.bits)),
					    0};
					bin.append(reinterpret_cast<const char*>(v), sizeof(v));
				}

				StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_16, false, 8};
				return format;
			}
		}
	}
	else if (stream.type == cgltf_attribute_type_texcoord)
	{
		if (!settings.quantize)
			return writeVertexStreamRaw(bin, stream, cgltf_type_vec2, 2);

		float uv_rscale[2] = {
		    qt.scale[0] == 0.f ? 0.f : 1.f / qt.scale[0],
		    qt.scale[1] == 0.f ? 0.f : 1.f / qt.scale[1],
		};

		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			uint16_t v[2] = {
			    uint16_t(meshopt_quantizeUnorm((a.f[0] - qt.offset[0]) * uv_rscale[0], qt.bits)),
			    uint16_t(meshopt_quantizeUnorm((a.f[1] - qt.offset[1]) * uv_rscale[1], qt.bits)),
			};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec2, cgltf_component_type_r_16u, false, 4};
		return format;
	}
	else if (stream.type == cgltf_attribute_type_normal)
	{
		if (!settings.quantize)
			return writeVertexStreamRaw(bin, stream, cgltf_type_vec3, 3);

		bool oct = settings.compressmore && stream.target == 0;
		int bits = settings.nrm_bits;

		StreamFormat::Filter filter = oct ? StreamFormat::Filter_Oct : StreamFormat::Filter_None;

		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			float nx = a.f[0], ny = a.f[1], nz = a.f[2];

			if (bits > 8)
			{
				int16_t v[4];

				if (oct)
				{
					int fu, fv;
					encodeOct(fu, fv, nx, ny, nz, bits);

					v[0] = int16_t(fu);
					v[1] = int16_t(fv);
					v[2] = int16_t(meshopt_quantizeSnorm(1.f, bits));
					v[3] = 0;
				}
				else
				{
					v[0] = int16_t(meshopt_quantizeSnorm(nx, bits));
					v[1] = int16_t(meshopt_quantizeSnorm(ny, bits));
					v[2] = int16_t(meshopt_quantizeSnorm(nz, bits));
					v[3] = 0;
				}

				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}
			else
			{
				int8_t v[4];

				if (oct)
				{
					int fu, fv;
					encodeOct(fu, fv, nx, ny, nz, bits);

					v[0] = int8_t(fu);
					v[1] = int8_t(fv);
					v[2] = int8_t(meshopt_quantizeSnorm(1.f, bits));
					v[3] = 0;
				}
				else
				{
					v[0] = int8_t(meshopt_quantizeSnorm(nx, bits));
					v[1] = int8_t(meshopt_quantizeSnorm(ny, bits));
					v[2] = int8_t(meshopt_quantizeSnorm(nz, bits));
					v[3] = 0;
				}

				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}
		}

		if (bits > 8)
		{
			StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_16, true, 8, filter};
			return format;
		}
		else
		{
			StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_8, true, 4, filter};
			return format;
		}
	}
	else if (stream.type == cgltf_attribute_type_tangent)
	{
		if (!settings.quantize)
			return writeVertexStreamRaw(bin, stream, cgltf_type_vec4, 4);

		bool oct = settings.compressmore && stream.target == 0;
		int bits = (settings.nrm_bits > 8) ? 8 : settings.nrm_bits;

		StreamFormat::Filter filter = oct ? StreamFormat::Filter_Oct : StreamFormat::Filter_None;

		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			float nx = a.f[0], ny = a.f[1], nz = a.f[2], nw = a.f[3];

			int8_t v[4];

			if (oct)
			{
				int fu, fv;
				encodeOct(fu, fv, nx, ny, nz, bits);

				v[0] = int8_t(fu);
				v[1] = int8_t(fv);
				v[2] = int8_t(meshopt_quantizeSnorm(1.f, bits));
				v[3] = int8_t(meshopt_quantizeSnorm(nw, bits));
			}
			else
			{
				v[0] = int8_t(meshopt_quantizeSnorm(nx, bits));
				v[1] = int8_t(meshopt_quantizeSnorm(ny, bits));
				v[2] = int8_t(meshopt_quantizeSnorm(nz, bits));
				v[3] = int8_t(meshopt_quantizeSnorm(nw, bits));
			}

			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		cgltf_type type = (stream.target == 0) ? cgltf_type_vec4 : cgltf_type_vec3;

		StreamFormat format = {type, cgltf_component_type_r_8, true, 4, filter};
		return format;
	}
	else if (stream.type == cgltf_attribute_type_color)
	{
		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			uint8_t v[4] = {
			    uint8_t(meshopt_quantizeUnorm(a.f[0], 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[1], 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[2], 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[3], 8))};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_8u, true, 4};
		return format;
	}
	else if (stream.type == cgltf_attribute_type_weights)
	{
		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			float ws = a.f[0] + a.f[1] + a.f[2] + a.f[3];
			float wsi = (ws == 0.f) ? 0.f : 1.f / ws;

			uint8_t v[4] = {
			    uint8_t(meshopt_quantizeUnorm(a.f[0] * wsi, 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[1] * wsi, 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[2] * wsi, 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[3] * wsi, 8))};

			if (wsi != 0.f)
				renormalizeWeights(v);

			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_8u, true, 4};
		return format;
	}
	else if (stream.type == cgltf_attribute_type_joints)
	{
		unsigned int maxj = 0;

		for (size_t i = 0; i < stream.data.size(); ++i)
			maxj = std::max(maxj, unsigned(stream.data[i].f[0]));

		assert(maxj <= 65535);

		if (maxj <= 255)
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				const Attr& a = stream.data[i];

				uint8_t v[4] = {
				    uint8_t(a.f[0]),
				    uint8_t(a.f[1]),
				    uint8_t(a.f[2]),
				    uint8_t(a.f[3])};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}

			StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_8u, false, 4};
			return format;
		}
		else
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				const Attr& a = stream.data[i];

				uint16_t v[4] = {
				    uint16_t(a.f[0]),
				    uint16_t(a.f[1]),
				    uint16_t(a.f[2]),
				    uint16_t(a.f[3])};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}

			StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_16u, false, 8};
			return format;
		}
	}
	else
	{
		return writeVertexStreamRaw(bin, stream, cgltf_type_vec4, 4);
	}
}

StreamFormat writeIndexStream(std::string& bin, const std::vector<unsigned int>& stream)
{
	unsigned int maxi = 0;
	for (size_t i = 0; i < stream.size(); ++i)
		maxi = std::max(maxi, stream[i]);

	// save 16-bit indices if we can; note that we can't use restart index (65535)
	if (maxi < 65535)
	{
		for (size_t i = 0; i < stream.size(); ++i)
		{
			uint16_t v[1] = {uint16_t(stream[i])};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_scalar, cgltf_component_type_r_16u, false, 2};
		return format;
	}
	else
	{
		for (size_t i = 0; i < stream.size(); ++i)
		{
			uint32_t v[1] = {stream[i]};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_scalar, cgltf_component_type_r_32u, false, 4};
		return format;
	}
}

StreamFormat writeTimeStream(std::string& bin, const std::vector<float>& data)
{
	for (size_t i = 0; i < data.size(); ++i)
	{
		float v[1] = {data[i]};
		bin.append(reinterpret_cast<const char*>(v), sizeof(v));
	}

	StreamFormat format = {cgltf_type_scalar, cgltf_component_type_r_32f, false, 4};
	return format;
}

static void encodeQuat(int16_t v[4], const Attr& a, int bits)
{
	const float scaler = sqrtf(2.f);

	// establish maximum quaternion component
	int qc = 0;
	qc = fabsf(a.f[1]) > fabsf(a.f[qc]) ? 1 : qc;
	qc = fabsf(a.f[2]) > fabsf(a.f[qc]) ? 2 : qc;
	qc = fabsf(a.f[3]) > fabsf(a.f[qc]) ? 3 : qc;

	// we use double-cover properties to discard the sign
	float sign = a.f[qc] < 0.f ? -1.f : 1.f;

	// note: we always encode a cyclical swizzle to be able to recover the order via rotation
	v[0] = int16_t(meshopt_quantizeSnorm(a.f[(qc + 1) & 3] * scaler * sign, bits));
	v[1] = int16_t(meshopt_quantizeSnorm(a.f[(qc + 2) & 3] * scaler * sign, bits));
	v[2] = int16_t(meshopt_quantizeSnorm(a.f[(qc + 3) & 3] * scaler * sign, bits));
	v[3] = int16_t((meshopt_quantizeSnorm(1.f, bits) & ~0xff) | qc);
}

StreamFormat writeKeyframeStream(std::string& bin, cgltf_animation_path_type type, const std::vector<Attr>& data, const Settings& settings, int optbits)
{
	if (type == cgltf_animation_path_type_rotation)
	{
		StreamFormat::Filter filter = settings.compressmore ? StreamFormat::Filter_Quat : StreamFormat::Filter_None;

		for (size_t i = 0; i < data.size(); ++i)
		{
			const Attr& a = data[i];

			int16_t v[4];

			if (filter == StreamFormat::Filter_Quat)
			{
				encodeQuat(v, a, optbits);
			}
			else
			{
				v[0] = int16_t(meshopt_quantizeSnorm(a.f[0], 16));
				v[1] = int16_t(meshopt_quantizeSnorm(a.f[1], 16));
				v[2] = int16_t(meshopt_quantizeSnorm(a.f[2], 16));
				v[3] = int16_t(meshopt_quantizeSnorm(a.f[3], 16));
			}

			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_16, true, 8, filter};
		return format;
	}
	else if (type == cgltf_animation_path_type_weights)
	{
		for (size_t i = 0; i < data.size(); ++i)
		{
			const Attr& a = data[i];

			uint8_t v[1] = {uint8_t(meshopt_quantizeUnorm(a.f[0], 8))};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_scalar, cgltf_component_type_r_8u, true, 1};
		return format;
	}
	else if (type == cgltf_animation_path_type_translation || type == cgltf_animation_path_type_scale)
	{
		// 9 bits for sign + exponent, the rest is for mantissa
		int mbits = (type == cgltf_animation_path_type_translation ? settings.trn_bits : settings.scl_bits) - 9;

		for (size_t i = 0; i < data.size(); ++i)
		{
			const Attr& a = data[i];

			float v[3] = {
			    meshopt_quantizeFloat(a.f[0], mbits),
			    meshopt_quantizeFloat(a.f[1], mbits),
			    meshopt_quantizeFloat(a.f[2], mbits)};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_32f, false, 12};
		return format;
	}
	else
	{
		for (size_t i = 0; i < data.size(); ++i)
		{
			const Attr& a = data[i];

			float v[4] = {a.f[0], a.f[1], a.f[2], a.f[3]};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_32f, false, 16};
		return format;
	}
}

void compressVertexStream(std::string& bin, const std::string& data, size_t count, size_t stride)
{
	assert(data.size() == count * stride);

	std::vector<unsigned char> compressed(meshopt_encodeVertexBufferBound(count, stride));
	size_t size = meshopt_encodeVertexBuffer(&compressed[0], compressed.size(), data.c_str(), count, stride);

	bin.append(reinterpret_cast<const char*>(&compressed[0]), size);
}

void compressIndexStream(std::string& bin, const std::string& data, size_t count, size_t stride)
{
	assert(stride == 2 || stride == 4);
	assert(data.size() == count * stride);

	std::vector<unsigned char> compressed(meshopt_encodeIndexBufferBound(count, count * 3));
	size_t size = 0;

	if (stride == 2)
		size = meshopt_encodeIndexBuffer(&compressed[0], compressed.size(), reinterpret_cast<const uint16_t*>(data.c_str()), count);
	else
		size = meshopt_encodeIndexBuffer(&compressed[0], compressed.size(), reinterpret_cast<const uint32_t*>(data.c_str()), count);

	bin.append(reinterpret_cast<const char*>(&compressed[0]), size);
}
