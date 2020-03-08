// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* componentType(cgltf_component_type type)
{
	switch (type)
	{
	case cgltf_component_type_r_8:
		return "5120";
	case cgltf_component_type_r_8u:
		return "5121";
	case cgltf_component_type_r_16:
		return "5122";
	case cgltf_component_type_r_16u:
		return "5123";
	case cgltf_component_type_r_32u:
		return "5125";
	case cgltf_component_type_r_32f:
		return "5126";
	default:
		return "0";
	}
}

static const char* shapeType(cgltf_type type)
{
	switch (type)
	{
	case cgltf_type_scalar:
		return "SCALAR";
	case cgltf_type_vec2:
		return "VEC2";
	case cgltf_type_vec3:
		return "VEC3";
	case cgltf_type_vec4:
		return "VEC4";
	case cgltf_type_mat2:
		return "MAT2";
	case cgltf_type_mat3:
		return "MAT3";
	case cgltf_type_mat4:
		return "MAT4";
	default:
		return "";
	}
}

const char* attributeType(cgltf_attribute_type type)
{
	switch (type)
	{
	case cgltf_attribute_type_position:
		return "POSITION";
	case cgltf_attribute_type_normal:
		return "NORMAL";
	case cgltf_attribute_type_tangent:
		return "TANGENT";
	case cgltf_attribute_type_texcoord:
		return "TEXCOORD";
	case cgltf_attribute_type_color:
		return "COLOR";
	case cgltf_attribute_type_joints:
		return "JOINTS";
	case cgltf_attribute_type_weights:
		return "WEIGHTS";
	default:
		return "ATTRIBUTE";
	}
}

const char* animationPath(cgltf_animation_path_type type)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		return "translation";
	case cgltf_animation_path_type_rotation:
		return "rotation";
	case cgltf_animation_path_type_scale:
		return "scale";
	case cgltf_animation_path_type_weights:
		return "weights";
	default:
		return "";
	}
}

static const char* lightType(cgltf_light_type type)
{
	switch (type)
	{
	case cgltf_light_type_directional:
		return "directional";
	case cgltf_light_type_point:
		return "point";
	case cgltf_light_type_spot:
		return "spot";
	default:
		return "";
	}
}

static void writeTextureInfo(std::string& json, const cgltf_data* data, const cgltf_texture_view& view, const QuantizationTexture* qt)
{
	assert(view.texture);

	cgltf_texture_transform transform = {};

	if (view.has_transform)
	{
		transform = view.transform;
	}
	else
	{
		transform.scale[0] = transform.scale[1] = 1.f;
	}

	if (qt)
	{
		transform.offset[0] += qt->offset[0];
		transform.offset[1] += qt->offset[1];
		transform.scale[0] *= qt->scale[0] / float((1 << qt->bits) - 1);
		transform.scale[1] *= qt->scale[1] / float((1 << qt->bits) - 1);
	}

	append(json, "{\"index\":");
	append(json, size_t(view.texture - data->textures));
	append(json, ",\"texCoord\":");
	append(json, size_t(view.texcoord));
	if (view.has_transform || qt)
	{
		append(json, ",\"extensions\":{\"KHR_texture_transform\":{");
		append(json, "\"offset\":[");
		append(json, transform.offset[0]);
		append(json, ",");
		append(json, transform.offset[1]);
		append(json, "],\"scale\":[");
		append(json, transform.scale[0]);
		append(json, ",");
		append(json, transform.scale[1]);
		append(json, "]");
		if (transform.rotation != 0.f)
		{
			append(json, ",\"rotation\":");
			append(json, transform.rotation);
		}
		append(json, "}}");
	}
	append(json, "}");
}

static const float white[4] = {1, 1, 1, 1};
static const float black[4] = {0, 0, 0, 0};

static void writeMaterialComponent(std::string& json, const cgltf_data* data, const cgltf_pbr_metallic_roughness& pbr, const QuantizationTexture* qt)
{
	comma(json);
	append(json, "\"pbrMetallicRoughness\":{");
	if (memcmp(pbr.base_color_factor, white, 16) != 0)
	{
		comma(json);
		append(json, "\"baseColorFactor\":[");
		append(json, pbr.base_color_factor[0]);
		append(json, ",");
		append(json, pbr.base_color_factor[1]);
		append(json, ",");
		append(json, pbr.base_color_factor[2]);
		append(json, ",");
		append(json, pbr.base_color_factor[3]);
		append(json, "]");
	}
	if (pbr.base_color_texture.texture)
	{
		comma(json);
		append(json, "\"baseColorTexture\":");
		writeTextureInfo(json, data, pbr.base_color_texture, qt);
	}
	if (pbr.metallic_factor != 1)
	{
		comma(json);
		append(json, "\"metallicFactor\":");
		append(json, pbr.metallic_factor);
	}
	if (pbr.roughness_factor != 1)
	{
		comma(json);
		append(json, "\"roughnessFactor\":");
		append(json, pbr.roughness_factor);
	}
	if (pbr.metallic_roughness_texture.texture)
	{
		comma(json);
		append(json, "\"metallicRoughnessTexture\":");
		writeTextureInfo(json, data, pbr.metallic_roughness_texture, qt);
	}
	append(json, "}");
}

static void writeMaterialComponent(std::string& json, const cgltf_data* data, const cgltf_pbr_specular_glossiness& pbr, const QuantizationTexture* qt)
{
	comma(json);
	append(json, "\"KHR_materials_pbrSpecularGlossiness\":{");
	if (pbr.diffuse_texture.texture)
	{
		comma(json);
		append(json, "\"diffuseTexture\":");
		writeTextureInfo(json, data, pbr.diffuse_texture, qt);
	}
	if (pbr.specular_glossiness_texture.texture)
	{
		comma(json);
		append(json, "\"specularGlossinessTexture\":");
		writeTextureInfo(json, data, pbr.specular_glossiness_texture, qt);
	}
	if (memcmp(pbr.diffuse_factor, white, 16) != 0)
	{
		comma(json);
		append(json, "\"diffuseFactor\":[");
		append(json, pbr.diffuse_factor[0]);
		append(json, ",");
		append(json, pbr.diffuse_factor[1]);
		append(json, ",");
		append(json, pbr.diffuse_factor[2]);
		append(json, ",");
		append(json, pbr.diffuse_factor[3]);
		append(json, "]");
	}
	if (memcmp(pbr.specular_factor, white, 12) != 0)
	{
		comma(json);
		append(json, "\"specularFactor\":[");
		append(json, pbr.specular_factor[0]);
		append(json, ",");
		append(json, pbr.specular_factor[1]);
		append(json, ",");
		append(json, pbr.specular_factor[2]);
		append(json, "]");
	}
	if (pbr.glossiness_factor != 1)
	{
		comma(json);
		append(json, "\"glossinessFactor\":");
		append(json, pbr.glossiness_factor);
	}
	append(json, "}");
}

static void writeMaterialComponent(std::string& json, const cgltf_data* data, const cgltf_clearcoat& cc, const QuantizationTexture* qt)
{
	comma(json);
	append(json, "\"KHR_materials_clearcoat\":{");
	if (cc.clearcoat_texture.texture)
	{
		comma(json);
		append(json, "\"clearcoatTexture\":");
		writeTextureInfo(json, data, cc.clearcoat_texture, qt);
	}
	if (cc.clearcoat_roughness_texture.texture)
	{
		comma(json);
		append(json, "\"clearcoatRoughnessTexture\":");
		writeTextureInfo(json, data, cc.clearcoat_roughness_texture, qt);
	}
	if (cc.clearcoat_normal_texture.texture)
	{
		comma(json);
		append(json, "\"clearcoatNormalTexture\":");
		writeTextureInfo(json, data, cc.clearcoat_normal_texture, qt);
	}
	if (cc.clearcoat_factor != 0)
	{
		comma(json);
		append(json, "\"clearcoatFactor\":");
		append(json, cc.clearcoat_factor);
	}
	if (cc.clearcoat_factor != 0)
	{
		comma(json);
		append(json, "\"clearcoatRoughnessFactor\":");
		append(json, cc.clearcoat_roughness_factor);
	}
	append(json, "}");
}

void writeMaterial(std::string& json, const cgltf_data* data, const cgltf_material& material, const QuantizationTexture* qt)
{
	if (material.name && *material.name)
	{
		comma(json);
		append(json, "\"name\":\"");
		append(json, material.name);
		append(json, "\"");
	}

	if (material.has_pbr_metallic_roughness)
	{
		writeMaterialComponent(json, data, material.pbr_metallic_roughness, qt);
	}

	if (material.normal_texture.texture)
	{
		comma(json);
		append(json, "\"normalTexture\":");
		writeTextureInfo(json, data, material.normal_texture, qt);
	}

	if (material.occlusion_texture.texture)
	{
		comma(json);
		append(json, "\"occlusionTexture\":");
		writeTextureInfo(json, data, material.occlusion_texture, qt);
	}

	if (material.emissive_texture.texture)
	{
		comma(json);
		append(json, "\"emissiveTexture\":");
		writeTextureInfo(json, data, material.emissive_texture, qt);
	}

	if (memcmp(material.emissive_factor, black, 12) != 0)
	{
		comma(json);
		append(json, "\"emissiveFactor\":[");
		append(json, material.emissive_factor[0]);
		append(json, ",");
		append(json, material.emissive_factor[1]);
		append(json, ",");
		append(json, material.emissive_factor[2]);
		append(json, "]");
	}

	if (material.alpha_mode != cgltf_alpha_mode_opaque)
	{
		comma(json);
		append(json, "\"alphaMode\":");
		append(json, (material.alpha_mode == cgltf_alpha_mode_blend) ? "\"BLEND\"" : "\"MASK\"");
	}

	if (material.alpha_cutoff != 0.5f)
	{
		comma(json);
		append(json, "\"alphaCutoff\":");
		append(json, material.alpha_cutoff);
	}

	if (material.double_sided)
	{
		comma(json);
		append(json, "\"doubleSided\":true");
	}

	if (material.has_pbr_specular_glossiness || material.has_clearcoat || material.unlit)
	{
		comma(json);
		append(json, "\"extensions\":{");

		if (material.has_pbr_specular_glossiness)
		{
			writeMaterialComponent(json, data, material.pbr_specular_glossiness, qt);
		}

		if (material.has_clearcoat)
		{
			writeMaterialComponent(json, data, material.clearcoat, qt);
		}

		if (material.unlit)
		{
			comma(json);
			append(json, "\"KHR_materials_unlit\":{}");
		}

		append(json, "}");
	}
}

size_t getBufferView(std::vector<BufferView>& views, BufferView::Kind kind, StreamFormat::Filter filter, int variant, size_t stride, bool compressed)
{
	if (variant >= 0)
	{
		for (size_t i = 0; i < views.size(); ++i)
		{
			BufferView& v = views[i];

			if (v.kind == kind && v.filter == filter && v.variant == variant && v.stride == stride && v.compressed == compressed)
				return i;
		}
	}

	BufferView view = {kind, filter, variant, stride, compressed};
	views.push_back(view);

	return views.size() - 1;
}

void writeBufferView(std::string& json, BufferView::Kind kind, StreamFormat::Filter filter, size_t count, size_t stride, size_t bin_offset, size_t bin_size, int compression, size_t compressed_offset, size_t compressed_size)
{
	assert(bin_size == count * stride);

	// when compression is enabled, we store uncompressed data in buffer 1 and compressed data in buffer 0
	// when compression is disabled, we store uncompressed data in buffer 0
	size_t buffer = compression >= 0 ? 1 : 0;

	append(json, "{\"buffer\":");
	append(json, buffer);
	append(json, ",\"byteOffset\":");
	append(json, bin_offset);
	append(json, ",\"byteLength\":");
	append(json, bin_size);
	if (kind == BufferView::Kind_Vertex)
	{
		append(json, ",\"byteStride\":");
		append(json, stride);
	}
	if (kind == BufferView::Kind_Vertex || kind == BufferView::Kind_Index)
	{
		append(json, ",\"target\":");
		append(json, (kind == BufferView::Kind_Vertex) ? "34962" : "34963");
	}
	if (compression >= 0)
	{
		append(json, ",\"extensions\":{");
		append(json, "\"MESHOPT_compression\":{");
		append(json, "\"buffer\":0");
		append(json, ",\"byteOffset\":");
		append(json, size_t(compressed_offset));
		append(json, ",\"byteLength\":");
		append(json, size_t(compressed_size));
		append(json, ",\"byteStride\":");
		append(json, stride);
		append(json, ",\"mode\":");
		append(json, size_t(compression));
		if (filter != StreamFormat::Filter_None)
		{
			append(json, ",\"filter\":");
			append(json, size_t(filter));
		}
		append(json, ",\"count\":");
		append(json, count);
		append(json, "}}");
	}
	append(json, "}");
}

static void writeAccessor(std::string& json, size_t view, size_t offset, cgltf_type type, cgltf_component_type component_type, bool normalized, size_t count, const float* min = 0, const float* max = 0, size_t numminmax = 0)
{
	append(json, "{\"bufferView\":");
	append(json, view);
	append(json, ",\"byteOffset\":");
	append(json, offset);
	append(json, ",\"componentType\":");
	append(json, componentType(component_type));
	append(json, ",\"count\":");
	append(json, count);
	append(json, ",\"type\":\"");
	append(json, shapeType(type));
	append(json, "\"");

	if (normalized)
	{
		append(json, ",\"normalized\":true");
	}

	if (min && max)
	{
		assert(numminmax);

		append(json, ",\"min\":[");
		for (size_t k = 0; k < numminmax; ++k)
		{
			comma(json);
			append(json, min[k]);
		}
		append(json, "],\"max\":[");
		for (size_t k = 0; k < numminmax; ++k)
		{
			comma(json);
			append(json, max[k]);
		}
		append(json, "]");
	}

	append(json, "}");
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

static void writeEmbeddedImage(std::string& json, std::vector<BufferView>& views, const char* data, size_t size, const char* mime_type)
{
	size_t view = getBufferView(views, BufferView::Kind_Image, StreamFormat::Filter_None, -1, 1, false);

	assert(views[view].data.empty());
	views[view].data.assign(data, size);

	append(json, "\"bufferView\":");
	append(json, view);
	append(json, ",\"mimeType\":\"");
	append(json, mime_type);
	append(json, "\"");
}

void writeImage(std::string& json, std::vector<BufferView>& views, const cgltf_image& image, const ImageInfo& info, size_t index, const char* input_path, const char* output_path, const Settings& settings)
{
	std::string img_data;
	std::string mime_type;

	if (image.uri && parseDataUri(image.uri, mime_type, img_data))
	{
		// we will re-embed img_data below
	}
	else if (image.buffer_view && image.buffer_view->buffer->data && image.mime_type)
	{
		const cgltf_buffer_view* view = image.buffer_view;

		img_data.assign(static_cast<const char*>(view->buffer->data) + view->offset, view->size);
		mime_type = image.mime_type;
	}
	else if (image.uri && settings.texture_embed)
	{
		std::string full_path = getFullPath(image.uri, input_path);

		if (!readFile(full_path.c_str(), img_data))
		{
			fprintf(stderr, "Warning: unable to read image %s, skipping\n", image.uri);
		}

		mime_type = inferMimeType(image.uri);
	}

	if (!img_data.empty())
	{
		if (settings.texture_basis)
		{
			std::string encoded;

			if (encodeBasis(img_data, encoded, info.normal_map, info.srgb, settings.texture_quality))
			{
				if (settings.texture_ktx2)
					encoded = basisToKtx(encoded, info.srgb);

				writeEmbeddedImage(json, views, encoded.c_str(), encoded.size(), settings.texture_ktx2 ? "image/ktx2" : "image/basis");
			}
			else
			{
				fprintf(stderr, "Warning: unable to encode image %d with Basis, skipping\n", int(index));
			}
		}
		else
		{
			writeEmbeddedImage(json, views, img_data.c_str(), img_data.size(), mime_type.c_str());
		}
	}
	else if (image.uri)
	{
		if (settings.texture_basis)
		{
			std::string full_path = getFullPath(image.uri, input_path);
			std::string basis_path = getFileName(image.uri) + (settings.texture_ktx2 ? ".ktx2" : ".basis");
			std::string basis_full_path = getFullPath(basis_path.c_str(), output_path);

			if (readFile(full_path.c_str(), img_data))
			{
				std::string encoded;

				if (encodeBasis(img_data, encoded, info.normal_map, info.srgb, settings.texture_quality))
				{
					if (settings.texture_ktx2)
						encoded = basisToKtx(encoded, info.srgb);

					if (writeFile(basis_full_path.c_str(), encoded))
					{
						append(json, "\"uri\":\"");
						append(json, basis_path);
						append(json, "\"");
					}
					else
					{
						fprintf(stderr, "Warning: unable to save Basis image %s, skipping\n", image.uri);
					}
				}
				else
				{
					fprintf(stderr, "Warning: unable to encode image %s with Basis, skipping\n", image.uri);
				}
			}
			else
			{
				fprintf(stderr, "Warning: unable to read image %s, skipping\n", image.uri);
			}
		}
		else
		{
			append(json, "\"uri\":\"");
			append(json, image.uri);
			append(json, "\"");
		}
	}
	else
	{
		fprintf(stderr, "Warning: ignoring image %d since it has no URI and no valid buffer data\n", int(index));
	}
}

void writeTexture(std::string& json, const cgltf_texture& texture, cgltf_data* data, const Settings& settings)
{
	if (texture.image)
	{
		if (settings.texture_ktx2)
		{
			append(json, "\"extensions\":{\"KHR_texture_basisu\":{\"source\":");
			append(json, size_t(texture.image - data->images));
			append(json, "}}");
		}
		else
		{
			append(json, "\"source\":");
			append(json, size_t(texture.image - data->images));
		}
	}
}

void writeMeshAttributes(std::string& json, std::vector<BufferView>& views, std::string& json_accessors, size_t& accr_offset, const Mesh& mesh, int target, const QuantizationPosition& qp, const QuantizationTexture& qt, const Settings& settings)
{
	std::string scratch;

	for (size_t j = 0; j < mesh.streams.size(); ++j)
	{
		const Stream& stream = mesh.streams[j];

		if (stream.target != target)
			continue;

		scratch.clear();
		StreamFormat format = writeVertexStream(scratch, stream, qp, qt, settings);

		size_t view = getBufferView(views, BufferView::Kind_Vertex, format.filter, stream.type, format.stride, settings.compress);
		size_t offset = views[view].data.size();
		views[view].data += scratch;

		comma(json_accessors);
		if (stream.type == cgltf_attribute_type_position)
		{
			float min[3] = {};
			float max[3] = {};
			getPositionBounds(min, max, stream, settings.quantize ? &qp : NULL);

			writeAccessor(json_accessors, view, offset, format.type, format.component_type, format.normalized, stream.data.size(), min, max, 3);
		}
		else
		{
			writeAccessor(json_accessors, view, offset, format.type, format.component_type, format.normalized, stream.data.size());
		}

		size_t vertex_accr = accr_offset++;

		comma(json);
		append(json, "\"");
		append(json, attributeType(stream.type));
		if (stream.type != cgltf_attribute_type_position && stream.type != cgltf_attribute_type_normal && stream.type != cgltf_attribute_type_tangent)
		{
			append(json, "_");
			append(json, size_t(stream.index));
		}
		append(json, "\":");
		append(json, vertex_accr);
	}
}

size_t writeMeshIndices(std::vector<BufferView>& views, std::string& json_accessors, size_t& accr_offset, const Mesh& mesh, const Settings& settings)
{
	std::string scratch;
	StreamFormat format = writeIndexStream(scratch, mesh.indices);

	size_t view = getBufferView(views, BufferView::Kind_Index, StreamFormat::Filter_None, 0, format.stride, settings.compress);
	size_t offset = views[view].data.size();
	views[view].data += scratch;

	comma(json_accessors);
	writeAccessor(json_accessors, view, offset, format.type, format.component_type, format.normalized, mesh.indices.size());

	size_t index_accr = accr_offset++;

	return index_accr;
}

static size_t writeAnimationTime(std::vector<BufferView>& views, std::string& json_accessors, size_t& accr_offset, float mint, int frames, const Settings& settings)
{
	std::vector<float> time(frames);

	for (int j = 0; j < frames; ++j)
		time[j] = mint + float(j) / settings.anim_freq;

	std::string scratch;
	StreamFormat format = writeTimeStream(scratch, time);

	size_t view = getBufferView(views, BufferView::Kind_Time, StreamFormat::Filter_None, 0, format.stride, settings.compress);
	size_t offset = views[view].data.size();
	views[view].data += scratch;

	comma(json_accessors);
	writeAccessor(json_accessors, view, offset, cgltf_type_scalar, format.component_type, format.normalized, frames, &time.front(), &time.back(), 1);

	size_t time_accr = accr_offset++;

	return time_accr;
}

size_t writeJointBindMatrices(std::vector<BufferView>& views, std::string& json_accessors, size_t& accr_offset, const cgltf_skin& skin, const QuantizationPosition& qp, const Settings& settings)
{
	std::string scratch;

	for (size_t j = 0; j < skin.joints_count; ++j)
	{
		float transform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

		if (skin.inverse_bind_matrices)
		{
			cgltf_accessor_read_float(skin.inverse_bind_matrices, j, transform, 16);
		}

		if (settings.quantize)
		{
			float node_scale = qp.scale / float((1 << qp.bits) - 1);

			// pos_offset has to be applied first, thus it results in an offset rotated by the bind matrix
			transform[12] += qp.offset[0] * transform[0] + qp.offset[1] * transform[4] + qp.offset[2] * transform[8];
			transform[13] += qp.offset[0] * transform[1] + qp.offset[1] * transform[5] + qp.offset[2] * transform[9];
			transform[14] += qp.offset[0] * transform[2] + qp.offset[1] * transform[6] + qp.offset[2] * transform[10];

			// node_scale will be applied before the rotation/scale from transform
			for (int k = 0; k < 12; ++k)
				transform[k] *= node_scale;
		}

		scratch.append(reinterpret_cast<const char*>(transform), sizeof(transform));
	}

	size_t view = getBufferView(views, BufferView::Kind_Skin, StreamFormat::Filter_None, 0, 64, settings.compress);
	size_t offset = views[view].data.size();
	views[view].data += scratch;

	comma(json_accessors);
	writeAccessor(json_accessors, view, offset, cgltf_type_mat4, cgltf_component_type_r_32f, false, skin.joints_count);

	size_t matrix_accr = accr_offset++;

	return matrix_accr;
}

void writeMeshNode(std::string& json, size_t mesh_offset, const Mesh& mesh, cgltf_data* data, const QuantizationPosition* qp)
{
	comma(json);
	append(json, "{\"mesh\":");
	append(json, mesh_offset);
	if (mesh.skin)
	{
		comma(json);
		append(json, "\"skin\":");
		append(json, size_t(mesh.skin - data->skins));
	}
	if (qp)
	{
		float node_scale = qp->scale / float((1 << qp->bits) - 1);

		append(json, ",\"translation\":[");
		append(json, qp->offset[0]);
		append(json, ",");
		append(json, qp->offset[1]);
		append(json, ",");
		append(json, qp->offset[2]);
		append(json, "],\"scale\":[");
		append(json, node_scale);
		append(json, ",");
		append(json, node_scale);
		append(json, ",");
		append(json, node_scale);
		append(json, "]");
	}
	if (mesh.node && mesh.node->weights_count)
	{
		append(json, ",\"weights\":[");
		for (size_t j = 0; j < mesh.node->weights_count; ++j)
		{
			comma(json);
			append(json, mesh.node->weights[j]);
		}
		append(json, "]");
	}
	append(json, "}");
}

void writeSkin(std::string& json, const cgltf_skin& skin, size_t matrix_accr, const std::vector<NodeInfo>& nodes, cgltf_data* data)
{
	comma(json);
	append(json, "{");
	append(json, "\"joints\":[");
	for (size_t j = 0; j < skin.joints_count; ++j)
	{
		comma(json);
		append(json, size_t(nodes[skin.joints[j] - data->nodes].remap));
	}
	append(json, "]");
	append(json, ",\"inverseBindMatrices\":");
	append(json, matrix_accr);
	if (skin.skeleton)
	{
		comma(json);
		append(json, "\"skeleton\":");
		append(json, size_t(nodes[skin.skeleton - data->nodes].remap));
	}
	append(json, "}");
}

void writeNode(std::string& json, const cgltf_node& node, const std::vector<NodeInfo>& nodes, cgltf_data* data)
{
	const NodeInfo& ni = nodes[&node - data->nodes];

	comma(json);
	append(json, "{");
	if (node.name && *node.name)
	{
		comma(json);
		append(json, "\"name\":\"");
		append(json, node.name);
		append(json, "\"");
	}
	if (node.has_translation)
	{
		comma(json);
		append(json, "\"translation\":[");
		append(json, node.translation[0]);
		append(json, ",");
		append(json, node.translation[1]);
		append(json, ",");
		append(json, node.translation[2]);
		append(json, "]");
	}
	if (node.has_rotation)
	{
		comma(json);
		append(json, "\"rotation\":[");
		append(json, node.rotation[0]);
		append(json, ",");
		append(json, node.rotation[1]);
		append(json, ",");
		append(json, node.rotation[2]);
		append(json, ",");
		append(json, node.rotation[3]);
		append(json, "]");
	}
	if (node.has_scale)
	{
		comma(json);
		append(json, "\"scale\":[");
		append(json, node.scale[0]);
		append(json, ",");
		append(json, node.scale[1]);
		append(json, ",");
		append(json, node.scale[2]);
		append(json, "]");
	}
	if (node.has_matrix)
	{
		comma(json);
		append(json, "\"matrix\":[");
		for (int k = 0; k < 16; ++k)
		{
			comma(json);
			append(json, node.matrix[k]);
		}
		append(json, "]");
	}
	if (node.children_count || !ni.meshes.empty())
	{
		comma(json);
		append(json, "\"children\":[");
		for (size_t j = 0; j < node.children_count; ++j)
		{
			const NodeInfo& ci = nodes[node.children[j] - data->nodes];

			if (ci.keep)
			{
				comma(json);
				append(json, size_t(ci.remap));
			}
		}
		for (size_t j = 0; j < ni.meshes.size(); ++j)
		{
			comma(json);
			append(json, ni.meshes[j]);
		}
		append(json, "]");
	}
	if (node.camera)
	{
		comma(json);
		append(json, "\"camera\":");
		append(json, size_t(node.camera - data->cameras));
	}
	if (node.light)
	{
		comma(json);
		append(json, "\"extensions\":{\"KHR_lights_punctual\":{\"light\":");
		append(json, size_t(node.light - data->lights));
		append(json, "}}");
	}
	append(json, "}");
}

void writeAnimation(std::string& json, std::vector<BufferView>& views, std::string& json_accessors, size_t& accr_offset, const Animation& animation, size_t i, cgltf_data* data, const std::vector<NodeInfo>& nodes, const Settings& settings)
{
	std::vector<const Track*> tracks;

	for (size_t j = 0; j < animation.tracks.size(); ++j)
	{
		const Track& track = animation.tracks[j];

		const NodeInfo& ni = nodes[track.node - data->nodes];

		if (!ni.keep)
			continue;

		if (!settings.anim_const && (ni.animated_paths & (1 << track.path)) == 0)
			continue;

		tracks.push_back(&track);
	}

	if (tracks.empty())
	{
		fprintf(stderr, "Warning: ignoring animation %d because it has no valid tracks\n", int(i));
		return;
	}

	bool needs_time = false;
	bool needs_pose = false;

	for (size_t j = 0; j < tracks.size(); ++j)
	{
		const Track& track = *tracks[j];

		bool tc = track.data.size() == track.components;

		needs_time = needs_time || !tc;
		needs_pose = needs_pose || tc;
	}

	size_t time_accr = needs_time ? writeAnimationTime(views, json_accessors, accr_offset, animation.start, animation.frames, settings) : 0;
	size_t pose_accr = needs_pose ? writeAnimationTime(views, json_accessors, accr_offset, animation.start, 1, settings) : 0;

	std::string json_samplers;
	std::string json_channels;

	size_t track_offset = 0;

	for (size_t j = 0; j < tracks.size(); ++j)
	{
		const Track& track = *tracks[j];

		bool tc = track.data.size() == track.components;

		int bits = (track.path == cgltf_animation_path_type_rotation) ? nodes[track.node - data->nodes].bits : 0;

		std::string scratch;
		StreamFormat format = writeKeyframeStream(scratch, track.path, track.data, settings, bits);

		size_t view = getBufferView(views, BufferView::Kind_Keyframe, format.filter, track.path, format.stride, settings.compress && track.path != cgltf_animation_path_type_weights);
		size_t offset = views[view].data.size();
		views[view].data += scratch;

		comma(json_accessors);
		writeAccessor(json_accessors, view, offset, format.type, format.component_type, format.normalized, track.data.size());

		size_t data_accr = accr_offset++;

		comma(json_samplers);
		append(json_samplers, "{\"input\":");
		append(json_samplers, tc ? pose_accr : time_accr);
		append(json_samplers, ",\"output\":");
		append(json_samplers, data_accr);
		append(json_samplers, "}");

		const NodeInfo& tni = nodes[track.node - data->nodes];
		size_t target_node = size_t(tni.remap);

		if (track.path == cgltf_animation_path_type_weights)
		{
			assert(tni.meshes.size() == 1);
			target_node = tni.meshes[0];
		}

		comma(json_channels);
		append(json_channels, "{\"sampler\":");
		append(json_channels, track_offset);
		append(json_channels, ",\"target\":{\"node\":");
		append(json_channels, target_node);
		append(json_channels, ",\"path\":\"");
		append(json_channels, animationPath(track.path));
		append(json_channels, "\"}}");

		track_offset++;
	}

	comma(json);
	append(json, "{");
	if (animation.name && *animation.name)
	{
		append(json, "\"name\":\"");
		append(json, animation.name);
		append(json, "\",");
	}
	append(json, "\"samplers\":[");
	append(json, json_samplers);
	append(json, "],\"channels\":[");
	append(json, json_channels);
	append(json, "]}");
}

void writeCamera(std::string& json, const cgltf_camera& camera)
{
	comma(json);
	append(json, "{");

	switch (camera.type)
	{
	case cgltf_camera_type_perspective:
		append(json, "\"type\":\"perspective\",\"perspective\":{");
		append(json, "\"yfov\":");
		append(json, camera.data.perspective.yfov);
		append(json, ",\"znear\":");
		append(json, camera.data.perspective.znear);
		if (camera.data.perspective.aspect_ratio != 0.f)
		{
			append(json, ",\"aspectRatio\":");
			append(json, camera.data.perspective.aspect_ratio);
		}
		if (camera.data.perspective.zfar != 0.f)
		{
			append(json, ",\"zfar\":");
			append(json, camera.data.perspective.zfar);
		}
		append(json, "}");
		break;

	case cgltf_camera_type_orthographic:
		append(json, "\"type\":\"orthographic\",\"orthographic\":{");
		append(json, "\"xmag\":");
		append(json, camera.data.orthographic.xmag);
		append(json, ",\"ymag\":");
		append(json, camera.data.orthographic.ymag);
		append(json, ",\"znear\":");
		append(json, camera.data.orthographic.znear);
		append(json, ",\"zfar\":");
		append(json, camera.data.orthographic.zfar);
		append(json, "}");
		break;

	default:
		fprintf(stderr, "Warning: skipping camera of unknown type\n");
	}

	append(json, "}");
}

void writeLight(std::string& json, const cgltf_light& light)
{
	comma(json);
	append(json, "{\"type\":\"");
	append(json, lightType(light.type));
	append(json, "\"");
	if (memcmp(light.color, white, 12) != 0)
	{
		comma(json);
		append(json, "\"color\":[");
		append(json, light.color[0]);
		append(json, ",");
		append(json, light.color[1]);
		append(json, ",");
		append(json, light.color[2]);
		append(json, "]");
	}
	if (light.intensity != 1.f)
	{
		comma(json);
		append(json, "\"intensity\":");
		append(json, light.intensity);
	}
	if (light.range != 0.f)
	{
		comma(json);
		append(json, "\"range\":");
		append(json, light.range);
	}
	if (light.type == cgltf_light_type_spot)
	{
		comma(json);
		append(json, "\"spot\":{");
		append(json, "\"innerConeAngle\":");
		append(json, light.spot_inner_cone_angle);
		append(json, ",\"outerConeAngle\":");
		append(json, light.spot_outer_cone_angle == 0.f ? 0.78539816339f : light.spot_outer_cone_angle);
		append(json, "}");
	}
	append(json, "}");
}

void writeArray(std::string& json, const char* name, const std::string& contents)
{
	if (contents.empty())
		return;

	comma(json);
	append(json, "\"");
	append(json, name);
	append(json, "\":[");
	append(json, contents);
	append(json, "]");
}

void writeExtensions(std::string& json, const ExtensionInfo* extensions, size_t count)
{
	bool used_extensions = false;
	bool required_extensions = false;

	for (size_t i = 0; i < count; ++i)
	{
		used_extensions |= extensions[i].used;
		required_extensions |= extensions[i].used && extensions[i].required;
	}

	if (used_extensions)
	{
		comma(json);
		append(json, "\"extensionsUsed\":[");
		for (size_t i = 0; i < count; ++i)
			if (extensions[i].used)
			{
				comma(json);
				append(json, "\"");
				append(json, extensions[i].name);
				append(json, "\"");
			}
		append(json, "]");
	}

	if (required_extensions)
	{
		comma(json);
		append(json, "\"extensionsRequired\":[");
		for (size_t i = 0; i < count; ++i)
			if (extensions[i].used && extensions[i].required)
			{
				comma(json);
				append(json, "\"");
				append(json, extensions[i].name);
				append(json, "\"");
			}
		append(json, "]");
	}
}

void writeExtras(std::string& json, const cgltf_data* data, const cgltf_extras& extras)
{
	if (extras.start_offset == extras.end_offset)
		return;

	comma(json);
	append(json, "\"extras\":");
	appendJson(json, data->json + extras.start_offset, data->json + extras.end_offset);
}
