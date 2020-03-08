// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <algorithm>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/meshoptimizer.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

std::string getVersion()
{
	char result[32];
	sprintf(result, "%d.%d", MESHOPTIMIZER_VERSION / 1000, (MESHOPTIMIZER_VERSION % 1000) / 10);
	return result;
}

static void finalizeBufferViews(std::string& json, std::vector<BufferView>& views, std::string& bin, std::string& fallback)
{
	for (size_t i = 0; i < views.size(); ++i)
	{
		BufferView& view = views[i];

		size_t bin_offset = bin.size();
		size_t fallback_offset = fallback.size();

		size_t count = view.data.size() / view.stride;

		int compression = -1;

		if (view.compressed)
		{
			if (view.kind == BufferView::Kind_Index)
			{
				compressIndexStream(bin, view.data, count, view.stride);
				compression = 1;
			}
			else
			{
				compressVertexStream(bin, view.data, count, view.stride);
				compression = 0;
			}

			fallback += view.data;
		}
		else
		{
			bin += view.data;
		}

		size_t raw_offset = (compression >= 0) ? fallback_offset : bin_offset;

		comma(json);
		writeBufferView(json, view.kind, view.filter, count, view.stride, raw_offset, view.data.size(), compression, bin_offset, bin.size() - bin_offset);

		// record written bytes for statistics
		view.bytes = bin.size() - bin_offset;

		// align each bufferView by 4 bytes
		bin.resize((bin.size() + 3) & ~3);
		fallback.resize((fallback.size() + 3) & ~3);
	}
}

static void printMeshStats(const std::vector<Mesh>& meshes, const char* name)
{
	size_t triangles = 0;
	size_t vertices = 0;

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		const Mesh& mesh = meshes[i];

		triangles += mesh.indices.size() / 3;
		vertices += mesh.streams.empty() ? 0 : mesh.streams[0].data.size();
	}

	printf("%s: %d triangles, %d vertices\n", name, int(triangles), int(vertices));
}

static void printSceneStats(const std::vector<BufferView>& views, const std::vector<Mesh>& meshes, size_t node_offset, size_t mesh_offset, size_t material_offset, size_t json_size, size_t bin_size)
{
	size_t bytes[BufferView::Kind_Count] = {};

	for (size_t i = 0; i < views.size(); ++i)
	{
		const BufferView& view = views[i];
		bytes[view.kind] += view.bytes;
	}

	printf("output: %d nodes, %d meshes (%d primitives), %d materials\n", int(node_offset), int(mesh_offset), int(meshes.size()), int(material_offset));
	printf("output: JSON %d bytes, buffers %d bytes\n", int(json_size), int(bin_size));
	printf("output: buffers: vertex %d bytes, index %d bytes, skin %d bytes, time %d bytes, keyframe %d bytes, image %d bytes\n",
	       int(bytes[BufferView::Kind_Vertex]), int(bytes[BufferView::Kind_Index]), int(bytes[BufferView::Kind_Skin]),
	       int(bytes[BufferView::Kind_Time]), int(bytes[BufferView::Kind_Keyframe]), int(bytes[BufferView::Kind_Image]));
}

static void printAttributeStats(const std::vector<BufferView>& views, BufferView::Kind kind, const char* name)
{
	for (size_t i = 0; i < views.size(); ++i)
	{
		const BufferView& view = views[i];

		if (view.kind != kind)
			continue;

		const char* variant = "unknown";

		switch (kind)
		{
		case BufferView::Kind_Vertex:
			variant = attributeType(cgltf_attribute_type(view.variant));
			break;

		case BufferView::Kind_Index:
			variant = "index";
			break;

		case BufferView::Kind_Keyframe:
			variant = animationPath(cgltf_animation_path_type(view.variant));
			break;

		default:;
		}

		size_t count = view.data.size() / view.stride;

		printf("stats: %s %s: compressed %d bytes (%.1f bits), raw %d bytes (%d bits)\n",
		       name,
		       variant,
		       int(view.bytes),
		       double(view.bytes) / double(count) * 8,
		       int(view.data.size()),
		       int(view.stride * 8));
	}
}

static void process(cgltf_data* data, const char* input_path, const char* output_path, std::vector<Mesh>& meshes, std::vector<Animation>& animations, const Settings& settings, std::string& json, std::string& bin, std::string& fallback)
{
	if (settings.verbose)
	{
		printf("input: %d nodes, %d meshes (%d primitives), %d materials, %d skins, %d animations\n",
		       int(data->nodes_count), int(data->meshes_count), int(meshes.size()), int(data->materials_count), int(data->skins_count), int(animations.size()));
		printMeshStats(meshes, "input");
	}

	for (size_t i = 0; i < animations.size(); ++i)
	{
		processAnimation(animations[i], settings);
	}

	std::vector<NodeInfo> nodes(data->nodes_count);

	markAnimated(data, nodes, animations);

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		Mesh& mesh = meshes[i];

		// note: when -kn is specified, we keep mesh-node attachment so that named nodes can be transformed
		if (mesh.node && !settings.keep_named)
		{
			NodeInfo& ni = nodes[mesh.node - data->nodes];

			// we transform all non-skinned non-animated meshes to world space
			// this makes sure that quantization doesn't introduce gaps if the original scene was watertight
			if (!ni.animated && !mesh.skin && mesh.targets == 0)
			{
				transformMesh(mesh, mesh.node);
				mesh.node = 0;
			}

			// skinned and animated meshes will be anchored to the same node that they used to be in
			// for animated meshes, this is important since they need to be transformed by the same animation
			// for skinned meshes, in theory this isn't important since the transform of the skinned node doesn't matter; in practice this affects generated bounding box in three.js
		}
	}

	mergeMeshMaterials(data, meshes, settings);
	mergeMeshes(meshes, settings);
	filterEmptyMeshes(meshes);

	markNeededNodes(data, nodes, meshes, animations, settings);

	std::vector<MaterialInfo> materials(data->materials_count);

	markNeededMaterials(data, materials, meshes);

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		processMesh(meshes[i], settings);
	}

	filterEmptyMeshes(meshes); // some meshes may become empty after processing

	if (1)
	{
		analyzeBoneRadius(data, nodes, meshes);

		for (size_t i = 0; i < animations.size(); ++i)
		{
			analyzeAnimation(data, nodes, animations[i]);
		}
	}

	std::vector<ImageInfo> images(data->images_count);

	analyzeImages(data, images);

	QuantizationPosition qp = prepareQuantizationPosition(meshes, settings);

	std::vector<QuantizationTexture> qt_materials(materials.size());
	prepareQuantizationTexture(data, qt_materials, meshes, settings);

	QuantizationTexture qt_dummy = {};
	qt_dummy.bits = settings.tex_bits;

	std::string json_images;
	std::string json_textures;
	std::string json_materials;
	std::string json_accessors;
	std::string json_meshes;
	std::string json_nodes;
	std::string json_skins;
	std::string json_roots;
	std::string json_animations;
	std::string json_cameras;
	std::string json_lights;

	std::vector<BufferView> views;

	bool ext_pbr_specular_glossiness = false;
	bool ext_clearcoat = false;
	bool ext_unlit = false;

	size_t accr_offset = 0;
	size_t node_offset = 0;
	size_t mesh_offset = 0;
	size_t material_offset = 0;

	for (size_t i = 0; i < data->images_count; ++i)
	{
		const cgltf_image& image = data->images[i];

		if (settings.verbose && settings.texture_basis)
		{
			const char* uri = image.uri;
			bool embedded = !uri || strncmp(uri, "data:", 5) == 0;

			printf("image %d (%s) is being encoded with Basis\n", int(i), embedded ? "embedded" : uri);
		}

		comma(json_images);
		append(json_images, "{");
		writeImage(json_images, views, image, images[i], i, input_path, output_path, settings);
		append(json_images, "}");
	}

	for (size_t i = 0; i < data->textures_count; ++i)
	{
		const cgltf_texture& texture = data->textures[i];

		comma(json_textures);
		append(json_textures, "{");
		writeTexture(json_textures, texture, data, settings);
		append(json_textures, "}");
	}

	for (size_t i = 0; i < data->materials_count; ++i)
	{
		MaterialInfo& mi = materials[i];

		if (!mi.keep)
			continue;

		const cgltf_material& material = data->materials[i];

		comma(json_materials);
		append(json_materials, "{");
		writeMaterial(json_materials, data, material, settings.quantize ? &qt_materials[i] : NULL);
		if (settings.keep_extras)
			writeExtras(json_materials, data, material.extras);
		append(json_materials, "}");

		mi.remap = int(material_offset);
		material_offset++;

		ext_pbr_specular_glossiness = ext_pbr_specular_glossiness || material.has_pbr_specular_glossiness;
		ext_clearcoat = ext_clearcoat || material.has_clearcoat;
		ext_unlit = ext_unlit || material.unlit;
	}

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		const Mesh& mesh = meshes[i];

		comma(json_meshes);
		append(json_meshes, "{\"primitives\":[");

		size_t pi = i;
		for (; pi < meshes.size(); ++pi)
		{
			const Mesh& prim = meshes[pi];

			if (prim.node != mesh.node || prim.skin != mesh.skin || prim.targets != mesh.targets)
				break;

			if (!compareMeshTargets(mesh, prim))
				break;

			const QuantizationTexture& qt = prim.material ? qt_materials[prim.material - data->materials] : qt_dummy;

			comma(json_meshes);
			append(json_meshes, "{\"attributes\":{");
			writeMeshAttributes(json_meshes, views, json_accessors, accr_offset, prim, 0, qp, qt, settings);
			append(json_meshes, "}");
			append(json_meshes, ",\"mode\":");
			append(json_meshes, size_t(prim.type));

			if (mesh.targets)
			{
				append(json_meshes, ",\"targets\":[");
				for (size_t j = 0; j < mesh.targets; ++j)
				{
					comma(json_meshes);
					append(json_meshes, "{");
					writeMeshAttributes(json_meshes, views, json_accessors, accr_offset, prim, int(1 + j), qp, qt, settings);
					append(json_meshes, "}");
				}
				append(json_meshes, "]");
			}

			if (!prim.indices.empty())
			{
				size_t index_accr = writeMeshIndices(views, json_accessors, accr_offset, prim, settings);

				append(json_meshes, ",\"indices\":");
				append(json_meshes, index_accr);
			}

			if (prim.material)
			{
				MaterialInfo& mi = materials[prim.material - data->materials];

				assert(mi.keep);
				append(json_meshes, ",\"material\":");
				append(json_meshes, size_t(mi.remap));
			}

			append(json_meshes, "}");
		}

		append(json_meshes, "]");

		if (mesh.target_weights.size())
		{
			append(json_meshes, ",\"weights\":[");
			for (size_t j = 0; j < mesh.target_weights.size(); ++j)
			{
				comma(json_meshes);
				append(json_meshes, mesh.target_weights[j]);
			}
			append(json_meshes, "]");
		}

		if (mesh.target_names.size())
		{
			append(json_meshes, ",\"extras\":{\"targetNames\":[");
			for (size_t j = 0; j < mesh.target_names.size(); ++j)
			{
				comma(json_meshes);
				append(json_meshes, "\"");
				append(json_meshes, mesh.target_names[j]);
				append(json_meshes, "\"");
			}
			append(json_meshes, "]}");
		}

		append(json_meshes, "}");

		writeMeshNode(json_nodes, mesh_offset, mesh, data, settings.quantize ? &qp : NULL);

		if (mesh.node)
		{
			NodeInfo& ni = nodes[mesh.node - data->nodes];

			assert(ni.keep);
			ni.meshes.push_back(node_offset);
		}
		else
		{
			comma(json_roots);
			append(json_roots, node_offset);
		}

		node_offset++;
		mesh_offset++;

		// skip all meshes that we've written in this iteration
		assert(pi > i);
		i = pi - 1;
	}

	remapNodes(data, nodes, node_offset);

	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		NodeInfo& ni = nodes[i];

		if (!ni.keep)
			continue;

		const cgltf_node& node = data->nodes[i];

		if (!node.parent)
		{
			comma(json_roots);
			append(json_roots, size_t(ni.remap));
		}

		writeNode(json_nodes, node, nodes, data);
	}

	for (size_t i = 0; i < data->skins_count; ++i)
	{
		const cgltf_skin& skin = data->skins[i];

		size_t matrix_accr = writeJointBindMatrices(views, json_accessors, accr_offset, skin, qp, settings);

		writeSkin(json_skins, skin, matrix_accr, nodes, data);
	}

	for (size_t i = 0; i < animations.size(); ++i)
	{
		const Animation& animation = animations[i];

		writeAnimation(json_animations, views, json_accessors, accr_offset, animation, i, data, nodes, settings);
	}

	for (size_t i = 0; i < data->cameras_count; ++i)
	{
		const cgltf_camera& camera = data->cameras[i];

		writeCamera(json_cameras, camera);
	}

	for (size_t i = 0; i < data->lights_count; ++i)
	{
		const cgltf_light& light = data->lights[i];

		writeLight(json_lights, light);
	}

	append(json, "\"asset\":{");
	append(json, "\"version\":\"2.0\",\"generator\":\"gltfpack ");
	append(json, getVersion());
	append(json, "\"");
	writeExtras(json, data, data->asset.extras);
	append(json, "}");

	const ExtensionInfo extensions[] = {
	    {"KHR_mesh_quantization", settings.quantize, true},
	    {"MESHOPT_compression", settings.compress, !settings.fallback},
	    {"KHR_texture_transform", settings.quantize && !json_textures.empty(), false},
	    {"KHR_materials_pbrSpecularGlossiness", ext_pbr_specular_glossiness, false},
	    {"KHR_materials_clearcoat", ext_clearcoat, false},
	    {"KHR_materials_unlit", ext_unlit, false},
	    {"KHR_lights_punctual", data->lights_count > 0, false},
	    {"KHR_texture_basisu", !json_textures.empty() && settings.texture_ktx2, true},
	};

	writeExtensions(json, extensions, sizeof(extensions) / sizeof(extensions[0]));

	std::string json_views;
	finalizeBufferViews(json_views, views, bin, fallback);

	writeArray(json, "bufferViews", json_views);
	writeArray(json, "accessors", json_accessors);
	writeArray(json, "images", json_images);
	writeArray(json, "textures", json_textures);
	writeArray(json, "materials", json_materials);
	writeArray(json, "meshes", json_meshes);
	writeArray(json, "skins", json_skins);
	writeArray(json, "animations", json_animations);
	writeArray(json, "nodes", json_nodes);

	if (!json_roots.empty())
	{
		append(json, ",\"scenes\":[");
		append(json, "{\"nodes\":[");
		append(json, json_roots);
		append(json, "]}]");
	}

	writeArray(json, "cameras", json_cameras);

	if (!json_lights.empty())
	{
		append(json, ",\"extensions\":{\"KHR_lights_punctual\":{\"lights\":[");
		append(json, json_lights);
		append(json, "]}}");
	}
	if (!json_roots.empty())
	{
		append(json, ",\"scene\":0");
	}

	if (settings.verbose)
	{
		printMeshStats(meshes, "output");
		printSceneStats(views, meshes, node_offset, mesh_offset, material_offset, json.size(), bin.size());
	}

	if (settings.verbose > 1)
	{
		printAttributeStats(views, BufferView::Kind_Vertex, "vertex");
		printAttributeStats(views, BufferView::Kind_Index, "index");
		printAttributeStats(views, BufferView::Kind_Keyframe, "keyframe");
	}
}

static void writeU32(FILE* out, uint32_t data)
{
	fwrite(&data, 4, 1, out);
}

static const char* getBaseName(const char* path)
{
	const char* slash = strrchr(path, '/');
	const char* backslash = strrchr(path, '\\');

	const char* rs = slash ? slash + 1 : path;
	const char* bs = backslash ? backslash + 1 : path;

	return std::max(rs, bs);
}

static std::string getBufferSpec(const char* bin_path, size_t bin_size, const char* fallback_path, size_t fallback_size, bool fallback_ref)
{
	std::string json;
	append(json, "\"buffers\":[");
	append(json, "{");
	if (bin_path)
	{
		append(json, "\"uri\":\"");
		append(json, bin_path);
		append(json, "\"");
	}
	comma(json);
	append(json, "\"byteLength\":");
	append(json, bin_size);
	append(json, "}");
	if (fallback_ref)
	{
		comma(json);
		append(json, "{");
		if (fallback_path)
		{
			append(json, "\"uri\":\"");
			append(json, fallback_path);
			append(json, "\"");
		}
		comma(json);
		append(json, "\"byteLength\":");
		append(json, fallback_size);
		append(json, ",\"extensions\":{");
		append(json, "\"MESHOPT_compression\":{");
		append(json, "\"fallback\":true");
		append(json, "}}");
		append(json, "}");
	}
	append(json, "]");

	return json;
}

int gltfpack(const char* input, const char* output, const Settings& settings)
{
	cgltf_data* data = 0;
	std::vector<Mesh> meshes;
	std::vector<Animation> animations;

	const char* iext = strrchr(input, '.');

	if (iext && (strcmp(iext, ".gltf") == 0 || strcmp(iext, ".GLTF") == 0 || strcmp(iext, ".glb") == 0 || strcmp(iext, ".GLB") == 0))
	{
		const char* error = 0;
		data = parseGltf(input, meshes, animations, &error);

		if (error)
		{
			fprintf(stderr, "Error loading %s: %s\n", input, error);
			return 2;
		}
	}
	else if (iext && (strcmp(iext, ".obj") == 0 || strcmp(iext, ".OBJ") == 0))
	{
		const char* error = 0;
		data = parseObj(input, meshes, &error);

		if (!data)
		{
			fprintf(stderr, "Error loading %s: %s\n", input, error);
			return 2;
		}
	}
	else
	{
		fprintf(stderr, "Error loading %s: unknown extension (expected .gltf or .glb or .obj)\n", input);
		return 2;
	}

	if (data->images_count && settings.texture_basis)
	{
		if (!checkBasis())
		{
			fprintf(stderr, "Error: basisu is not present in PATH or BASISU_PATH is not set\n");
			return 3;
		}
	}

	std::string json, bin, fallback;
	process(data, input, output, meshes, animations, settings, json, bin, fallback);

	cgltf_free(data);

	if (!output)
	{
		return 0;
	}

	const char* oext = strrchr(output, '.');

	if (oext && (strcmp(oext, ".gltf") == 0 || strcmp(oext, ".GLTF") == 0))
	{
		std::string binpath = output;
		binpath.replace(binpath.size() - 5, 5, ".bin");

		std::string fbpath = output;
		fbpath.replace(fbpath.size() - 5, 5, ".fallback.bin");

		FILE* outjson = fopen(output, "wb");
		FILE* outbin = fopen(binpath.c_str(), "wb");
		FILE* outfb = settings.fallback ? fopen(fbpath.c_str(), "wb") : NULL;
		if (!outjson || !outbin || (!outfb && settings.fallback))
		{
			fprintf(stderr, "Error saving %s\n", output);
			return 4;
		}

		std::string bufferspec = getBufferSpec(getBaseName(binpath.c_str()), bin.size(), settings.fallback ? getBaseName(fbpath.c_str()) : NULL, fallback.size(), settings.compress);

		fprintf(outjson, "{");
		fwrite(bufferspec.c_str(), bufferspec.size(), 1, outjson);
		fprintf(outjson, ",");
		fwrite(json.c_str(), json.size(), 1, outjson);
		fprintf(outjson, "}");

		fwrite(bin.c_str(), bin.size(), 1, outbin);

		if (settings.fallback)
			fwrite(fallback.c_str(), fallback.size(), 1, outfb);

		fclose(outjson);
		fclose(outbin);
		if (outfb)
			fclose(outfb);
	}
	else if (oext && (strcmp(oext, ".glb") == 0 || strcmp(oext, ".GLB") == 0))
	{
		std::string fbpath = output;
		fbpath.replace(fbpath.size() - 4, 4, ".fallback.bin");

		FILE* out = fopen(output, "wb");
		FILE* outfb = settings.fallback ? fopen(fbpath.c_str(), "wb") : NULL;
		if (!out || (!outfb && settings.fallback))
		{
			fprintf(stderr, "Error saving %s\n", output);
			return 4;
		}

		std::string bufferspec = getBufferSpec(NULL, bin.size(), settings.fallback ? getBaseName(fbpath.c_str()) : NULL, fallback.size(), settings.compress);

		json.insert(0, "{" + bufferspec + ",");
		json.push_back('}');

		while (json.size() % 4)
			json.push_back(' ');

		while (bin.size() % 4)
			bin.push_back('\0');

		writeU32(out, 0x46546C67);
		writeU32(out, 2);
		writeU32(out, uint32_t(12 + 8 + json.size() + 8 + bin.size()));

		writeU32(out, uint32_t(json.size()));
		writeU32(out, 0x4E4F534A);
		fwrite(json.c_str(), json.size(), 1, out);

		writeU32(out, uint32_t(bin.size()));
		writeU32(out, 0x004E4942);
		fwrite(bin.c_str(), bin.size(), 1, out);

		if (settings.fallback)
			fwrite(fallback.c_str(), fallback.size(), 1, outfb);

		fclose(out);
		if (outfb)
			fclose(outfb);
	}
	else
	{
		fprintf(stderr, "Error saving %s: unknown extension (expected .gltf or .glb)\n", output);
		return 4;
	}

	return 0;
}

int main(int argc, char** argv)
{
	meshopt_encodeIndexVersion(1);

	Settings settings = {};
	settings.quantize = true;
	settings.pos_bits = 14;
	settings.tex_bits = 12;
	settings.nrm_bits = 8;
	settings.trn_bits = 24;
	settings.rot_bits = 12;
	settings.scl_bits = 24;
	settings.anim_freq = 30;
	settings.simplify_threshold = 1.f;
	settings.texture_quality = 50;

	const char* input = 0;
	const char* output = 0;
	bool help = false;
	bool test = false;

	for (int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];

		if (strcmp(arg, "-vp") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.pos_bits = atoi(argv[++i]);
		}
		else if (strcmp(arg, "-vt") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.tex_bits = atoi(argv[++i]);
		}
		else if (strcmp(arg, "-vn") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.nrm_bits = atoi(argv[++i]);
		}
		else if (strcmp(arg, "-at") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.trn_bits = atoi(argv[++i]);
		}
		else if (strcmp(arg, "-ar") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.rot_bits = atoi(argv[++i]);
		}
		else if (strcmp(arg, "-as") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.scl_bits = atoi(argv[++i]);
		}
		else if (strcmp(arg, "-af") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.anim_freq = atoi(argv[++i]);
		}
		else if (strcmp(arg, "-ac") == 0)
		{
			settings.anim_const = true;
		}
		else if (strcmp(arg, "-kn") == 0)
		{
			settings.keep_named = true;
		}
		else if (strcmp(arg, "-ke") == 0)
		{
			settings.keep_extras = true;
		}
		else if (strcmp(arg, "-si") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.simplify_threshold = float(atof(argv[++i]));
		}
		else if (strcmp(arg, "-sa") == 0)
		{
			settings.simplify_aggressive = true;
		}
		else if (strcmp(arg, "-te") == 0)
		{
			settings.texture_embed = true;
		}
		else if (strcmp(arg, "-tb") == 0)
		{
			settings.texture_basis = true;
		}
		else if (strcmp(arg, "-tc") == 0)
		{
			settings.texture_basis = true;
			settings.texture_ktx2 = true;
		}
		else if (strcmp(arg, "-tq") == 0 && i + 1 < argc && isdigit(argv[i + 1][0]))
		{
			settings.texture_quality = atoi(argv[++i]);
		}
		else if (strcmp(arg, "-noq") == 0)
		{
			settings.quantize = false;
		}
		else if (strcmp(arg, "-i") == 0 && i + 1 < argc && !input)
		{
			input = argv[++i];
		}
		else if (strcmp(arg, "-o") == 0 && i + 1 < argc && !output)
		{
			output = argv[++i];
		}
		else if (strcmp(arg, "-c") == 0)
		{
			settings.compress = true;
		}
		else if (strcmp(arg, "-cc") == 0)
		{
			settings.compress = true;
			settings.compressmore = true;
		}
		else if (strcmp(arg, "-cf") == 0)
		{
			settings.compress = true;
			settings.fallback = true;
		}
		else if (strcmp(arg, "-v") == 0)
		{
			settings.verbose = 1;
		}
		else if (strcmp(arg, "-vv") == 0)
		{
			settings.verbose = 2;
		}
		else if (strcmp(arg, "-h") == 0)
		{
			help = true;
		}
		else if (strcmp(arg, "-test") == 0)
		{
			test = true;
		}
		else if (arg[0] == '-')
		{
			fprintf(stderr, "Unrecognized option %s\n", arg);
			return 1;
		}
		else if (!test)
		{
			fprintf(stderr, "Expected option, got %s instead\n", arg);
			return 1;
		}
	}

	// shortcut for gltfpack -v
	if (settings.verbose && argc == 2)
	{
		printf("gltfpack %s\n", getVersion().c_str());
		return 0;
	}

	if (test)
	{
		for (int i = 1; i < argc; ++i)
		{
			const char* path = argv[i];
			if (path[0] == '-')
				continue;

			printf("%s\n", path);
			gltfpack(path, NULL, settings);
		}

		return 0;
	}

	if (!input || !output || help)
	{
		fprintf(stderr, "gltfpack %s\n", getVersion().c_str());
		fprintf(stderr, "Usage: gltfpack [options] -i input -o output\n");

		if (help)
		{
			fprintf(stderr, "\nBasics:\n");
			fprintf(stderr, "\t-i file: input file to process, .obj/.gltf/.glb\n");
			fprintf(stderr, "\t-o file: output file path, .gltf/.glb\n");
			fprintf(stderr, "\t-c: produce compressed gltf/glb files (-cc for higher compression ratio)\n");
			fprintf(stderr, "\nTextures:\n");
			fprintf(stderr, "\t-te: embed all textures into main buffer (.bin or .glb)\n");
			fprintf(stderr, "\t-tb: convert all textures to Basis Universal format (with basisu executable); will be removed in the future\n");
			fprintf(stderr, "\t-tc: convert all textures to KTX2 with BasisU supercompression (using basisu executable)\n");
			fprintf(stderr, "\t-tq N: set texture encoding quality (default: 50; N should be between 1 and 100\n");
			fprintf(stderr, "\nSimplification:\n");
			fprintf(stderr, "\t-si R: simplify meshes to achieve the ratio R (default: 1; R should be between 0 and 1)\n");
			fprintf(stderr, "\t-sa: aggressively simplify to the target ratio disregarding quality\n");
			fprintf(stderr, "\nVertices:\n");
			fprintf(stderr, "\t-vp N: use N-bit quantization for positions (default: 14; N should be between 1 and 16)\n");
			fprintf(stderr, "\t-vt N: use N-bit quantization for texture corodinates (default: 12; N should be between 1 and 16)\n");
			fprintf(stderr, "\t-vn N: use N-bit quantization for normals and tangents (default: 8; N should be between 1 and 16)\n");
			fprintf(stderr, "\nAnimations:\n");
			fprintf(stderr, "\t-at N: use N-bit quantization for translations (default: 24; N should be between 9 and 32)\n");
			fprintf(stderr, "\t-ar N: use N-bit quantization for rotations (default: 12; N should be between 9 and 16)\n");
			fprintf(stderr, "\t-as N: use N-bit quantization for scale (default: 24; N should be between 9 and 32)\n");
			fprintf(stderr, "\t-af N: resample animations at N Hz (default: 30)\n");
			fprintf(stderr, "\t-ac: keep constant animation tracks even if they don't modify the node transform\n");
			fprintf(stderr, "\nScene:\n");
			fprintf(stderr, "\t-kn: keep named nodes and meshes attached to named nodes so that named nodes can be transformed externally\n");
			fprintf(stderr, "\t-ke: keep extras data\n");
			fprintf(stderr, "\nMiscellaneous:\n");
			fprintf(stderr, "\t-cf: produce compressed gltf/glb files with fallback for loaders that don't support compression\n");
			fprintf(stderr, "\t-noq: disable quantization; produces much larger glTF files with no extensions\n");
			fprintf(stderr, "\t-v: verbose output (print version when used without other options)\n");
			fprintf(stderr, "\t-h: display this help and exit\n");
		}
		else
		{
			fprintf(stderr, "\nBasics:\n");
			fprintf(stderr, "\t-i file: input file to process, .obj/.gltf/.glb\n");
			fprintf(stderr, "\t-o file: output file path, .gltf/.glb\n");
			fprintf(stderr, "\t-c: produce compressed gltf/glb files (-cc for higher compression ratio)\n");
			fprintf(stderr, "\t-te: embed all textures into main buffer (.bin or .glb)\n");
			fprintf(stderr, "\t-tc: convert all textures to KTX2 with BasisU supercompression (using basisu executable)\n");
			fprintf(stderr, "\t-si R: simplify meshes to achieve the ratio R (default: 1; R should be between 0 and 1)\n");
			fprintf(stderr, "\nRun gltfpack -h to display a full list of options\n");
		}

		return 1;
	}

	return gltfpack(input, output, settings);
}
