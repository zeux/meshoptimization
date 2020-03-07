// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <algorithm>

#include <math.h>
#include <string.h>

#include "../src/meshoptimizer.h"

static void transformPosition(float* ptr, const float* transform)
{
	float x = ptr[0] * transform[0] + ptr[1] * transform[4] + ptr[2] * transform[8] + transform[12];
	float y = ptr[0] * transform[1] + ptr[1] * transform[5] + ptr[2] * transform[9] + transform[13];
	float z = ptr[0] * transform[2] + ptr[1] * transform[6] + ptr[2] * transform[10] + transform[14];

	ptr[0] = x;
	ptr[1] = y;
	ptr[2] = z;
}

static void transformNormal(float* ptr, const float* transform)
{
	float x = ptr[0] * transform[0] + ptr[1] * transform[4] + ptr[2] * transform[8];
	float y = ptr[0] * transform[1] + ptr[1] * transform[5] + ptr[2] * transform[9];
	float z = ptr[0] * transform[2] + ptr[1] * transform[6] + ptr[2] * transform[10];

	float l = sqrtf(x * x + y * y + z * z);
	float s = (l == 0.f) ? 0.f : 1 / l;

	ptr[0] = x * s;
	ptr[1] = y * s;
	ptr[2] = z * s;
}

void transformMesh(Mesh& mesh, const cgltf_node* node)
{
	float transform[16];
	cgltf_node_transform_world(node, transform);

	for (size_t si = 0; si < mesh.streams.size(); ++si)
	{
		Stream& stream = mesh.streams[si];

		if (stream.type == cgltf_attribute_type_position)
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
				transformPosition(stream.data[i].f, transform);
		}
		else if (stream.type == cgltf_attribute_type_normal || stream.type == cgltf_attribute_type_tangent)
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
				transformNormal(stream.data[i].f, transform);
		}
	}
}

bool compareMeshTargets(const Mesh& lhs, const Mesh& rhs)
{
	if (lhs.targets != rhs.targets)
		return false;

	if (lhs.target_weights.size() != rhs.target_weights.size())
		return false;

	for (size_t i = 0; i < lhs.target_weights.size(); ++i)
		if (lhs.target_weights[i] != rhs.target_weights[i])
			return false;

	if (lhs.target_names.size() != rhs.target_names.size())
		return false;

	for (size_t i = 0; i < lhs.target_names.size(); ++i)
		if (strcmp(lhs.target_names[i], rhs.target_names[i]) != 0)
			return false;

	return true;
}

static bool canMergeMeshes(const Mesh& lhs, const Mesh& rhs, const Settings& settings)
{
	if (lhs.node != rhs.node)
	{
		if (!lhs.node || !rhs.node)
			return false;

		if (lhs.node->parent != rhs.node->parent)
			return false;

		bool lhs_transform = lhs.node->has_translation | lhs.node->has_rotation | lhs.node->has_scale | lhs.node->has_matrix | (!!lhs.node->weights);
		bool rhs_transform = rhs.node->has_translation | rhs.node->has_rotation | rhs.node->has_scale | rhs.node->has_matrix | (!!rhs.node->weights);

		if (lhs_transform || rhs_transform)
			return false;

		if (settings.keep_named)
		{
			if (lhs.node->name && *lhs.node->name)
				return false;

			if (rhs.node->name && *rhs.node->name)
				return false;
		}

		// we can merge nodes that don't have transforms of their own and have the same parent
		// this is helpful when instead of splitting mesh into primitives, DCCs split mesh into mesh nodes
	}

	if (lhs.material != rhs.material)
		return false;

	if (lhs.skin != rhs.skin)
		return false;

	if (lhs.type != rhs.type)
		return false;

	if (!compareMeshTargets(lhs, rhs))
		return false;

	if (lhs.indices.empty() != rhs.indices.empty())
		return false;

	if (lhs.streams.size() != rhs.streams.size())
		return false;

	for (size_t i = 0; i < lhs.streams.size(); ++i)
		if (lhs.streams[i].type != rhs.streams[i].type || lhs.streams[i].index != rhs.streams[i].index || lhs.streams[i].target != rhs.streams[i].target)
			return false;

	return true;
}

static void mergeMeshes(Mesh& target, const Mesh& mesh)
{
	assert(target.streams.size() == mesh.streams.size());

	size_t vertex_offset = target.streams[0].data.size();
	size_t index_offset = target.indices.size();

	for (size_t i = 0; i < target.streams.size(); ++i)
		target.streams[i].data.insert(target.streams[i].data.end(), mesh.streams[i].data.begin(), mesh.streams[i].data.end());

	target.indices.resize(target.indices.size() + mesh.indices.size());

	size_t index_count = mesh.indices.size();

	for (size_t i = 0; i < index_count; ++i)
		target.indices[index_offset + i] = unsigned(vertex_offset + mesh.indices[i]);
}

void mergeMeshes(std::vector<Mesh>& meshes, const Settings& settings)
{
	for (size_t i = 0; i < meshes.size(); ++i)
	{
		Mesh& target = meshes[i];

		if (target.streams.empty())
			continue;

		size_t target_vertices = target.streams[0].data.size();
		size_t target_indices = target.indices.size();

		size_t last_merged = i;

		for (size_t j = i + 1; j < meshes.size(); ++j)
		{
			Mesh& mesh = meshes[j];

			if (!mesh.streams.empty() && canMergeMeshes(target, mesh, settings))
			{
				target_vertices += mesh.streams[0].data.size();
				target_indices += mesh.indices.size();
				last_merged = j;
			}
		}

		for (size_t j = 0; j < target.streams.size(); ++j)
			target.streams[j].data.reserve(target_vertices);

		target.indices.reserve(target_indices);

		for (size_t j = i + 1; j <= last_merged; ++j)
		{
			Mesh& mesh = meshes[j];

			if (!mesh.streams.empty() && canMergeMeshes(target, mesh, settings))
			{
				mergeMeshes(target, mesh);

				mesh.streams.clear();
				mesh.indices.clear();
			}
		}

		assert(target.streams[0].data.size() == target_vertices);
		assert(target.indices.size() == target_indices);
	}
}

void filterEmptyMeshes(std::vector<Mesh>& meshes)
{
	size_t write = 0;

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		Mesh& mesh = meshes[i];

		if (mesh.streams.empty())
			continue;

		if (mesh.streams[0].data.empty())
			continue;

		if (mesh.type == cgltf_primitive_type_triangles && mesh.indices.empty())
			continue;

		// the following code is roughly equivalent to meshes[write] = std::move(mesh)
		std::vector<Stream> streams;
		streams.swap(mesh.streams);

		std::vector<unsigned int> indices;
		indices.swap(mesh.indices);

		meshes[write] = mesh;
		meshes[write].streams.swap(streams);
		meshes[write].indices.swap(indices);

		write++;
	}

	meshes.resize(write);
}

static bool hasColors(const std::vector<Attr>& data)
{
	const float threshold = 0.99f;

	for (size_t i = 0; i < data.size(); ++i)
	{
		const Attr& a = data[i];

		if (a.f[0] < threshold || a.f[1] < threshold || a.f[2] < threshold || a.f[3] < threshold)
			return true;
	}

	return false;
}

static bool hasDeltas(const std::vector<Attr>& data)
{
	const float threshold = 0.01f;

	for (size_t i = 0; i < data.size(); ++i)
	{
		const Attr& a = data[i];

		if (fabsf(a.f[0]) > threshold || fabsf(a.f[1]) > threshold || fabsf(a.f[2]) > threshold)
			return true;
	}

	return false;
}

static void filterStreams(Mesh& mesh)
{
	bool morph_normal = false;
	bool morph_tangent = false;

	for (size_t i = 0; i < mesh.streams.size(); ++i)
	{
		Stream& stream = mesh.streams[i];

		if (stream.target)
		{
			morph_normal = morph_normal || (stream.type == cgltf_attribute_type_normal && hasDeltas(stream.data));
			morph_tangent = morph_tangent || (stream.type == cgltf_attribute_type_tangent && hasDeltas(stream.data));
		}
	}

	size_t write = 0;

	for (size_t i = 0; i < mesh.streams.size(); ++i)
	{
		Stream& stream = mesh.streams[i];

		if (stream.type == cgltf_attribute_type_texcoord && (!mesh.material || !usesTextureSet(*mesh.material, stream.index)))
			continue;

		if (stream.type == cgltf_attribute_type_tangent && (!mesh.material || !mesh.material->normal_texture.texture))
			continue;

		if ((stream.type == cgltf_attribute_type_joints || stream.type == cgltf_attribute_type_weights) && !mesh.skin)
			continue;

		if (stream.type == cgltf_attribute_type_color && !hasColors(stream.data))
			continue;

		if (stream.target && stream.type == cgltf_attribute_type_normal && !morph_normal)
			continue;

		if (stream.target && stream.type == cgltf_attribute_type_tangent && !morph_tangent)
			continue;

		// the following code is roughly equivalent to streams[write] = std::move(stream)
		std::vector<Attr> data;
		data.swap(stream.data);

		mesh.streams[write] = stream;
		mesh.streams[write].data.swap(data);

		write++;
	}

	mesh.streams.resize(write);
}

static void reindexMesh(Mesh& mesh)
{
	size_t total_vertices = mesh.streams[0].data.size();
	size_t total_indices = mesh.indices.size();

	std::vector<meshopt_Stream> streams;
	for (size_t i = 0; i < mesh.streams.size(); ++i)
	{
		if (mesh.streams[i].target)
			continue;

		assert(mesh.streams[i].data.size() == total_vertices);

		meshopt_Stream stream = {&mesh.streams[i].data[0], sizeof(Attr), sizeof(Attr)};
		streams.push_back(stream);
	}

	std::vector<unsigned int> remap(total_vertices);
	size_t unique_vertices = meshopt_generateVertexRemapMulti(&remap[0], &mesh.indices[0], total_indices, total_vertices, &streams[0], streams.size());
	assert(unique_vertices <= total_vertices);

	meshopt_remapIndexBuffer(&mesh.indices[0], &mesh.indices[0], total_indices, &remap[0]);

	for (size_t i = 0; i < mesh.streams.size(); ++i)
	{
		assert(mesh.streams[i].data.size() == total_vertices);

		meshopt_remapVertexBuffer(&mesh.streams[i].data[0], &mesh.streams[i].data[0], total_vertices, sizeof(Attr), &remap[0]);
		mesh.streams[i].data.resize(unique_vertices);
	}
}

static void filterTriangles(Mesh& mesh)
{
	unsigned int* indices = &mesh.indices[0];
	size_t total_indices = mesh.indices.size();

	size_t write = 0;

	for (size_t i = 0; i < total_indices; i += 3)
	{
		unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];

		if (a != b && a != c && b != c)
		{
			indices[write + 0] = a;
			indices[write + 1] = b;
			indices[write + 2] = c;
			write += 3;
		}
	}

	mesh.indices.resize(write);
}

static const Stream* getStream(const Mesh& mesh, cgltf_attribute_type type, int index = 0)
{
	for (size_t i = 0; i < mesh.streams.size(); ++i)
		if (mesh.streams[i].type == type && mesh.streams[i].index == index && mesh.streams[i].target == 0)
			return &mesh.streams[i];

	return 0;
}

static Stream* getStream(Mesh& mesh, cgltf_attribute_type type, int index = 0)
{
	return const_cast<Stream*>(getStream(static_cast<const Mesh&>(mesh), type, index));
}

static void simplifyMesh(Mesh& mesh, float threshold, bool aggressive)
{
	if (threshold >= 1)
		return;

	const Stream* positions = getStream(mesh, cgltf_attribute_type_position);
	if (!positions)
		return;

	size_t vertex_count = mesh.streams[0].data.size();

	size_t target_index_count = size_t(double(mesh.indices.size() / 3) * threshold) * 3;
	float target_error = 1e-2f;

	if (target_index_count < 1)
		return;

	std::vector<unsigned int> indices(mesh.indices.size());
	indices.resize(meshopt_simplify(&indices[0], &mesh.indices[0], mesh.indices.size(), positions->data[0].f, vertex_count, sizeof(Attr), target_index_count, target_error));
	mesh.indices.swap(indices);

	// Note: if the simplifier got stuck, we can try to reindex without normals/tangents and retry
	// For now we simply fall back to aggressive simplifier instead

	// if the mesh is complex enough and the precise simplifier got "stuck", we'll try to simplify using the sloppy simplifier which is guaranteed to reach the target count
	if (aggressive && target_index_count > 50 * 3 && mesh.indices.size() > target_index_count)
	{
		indices.resize(meshopt_simplifySloppy(&indices[0], &mesh.indices[0], mesh.indices.size(), positions->data[0].f, vertex_count, sizeof(Attr), target_index_count));
		mesh.indices.swap(indices);
	}
}

static void optimizeMesh(Mesh& mesh, bool compressmore)
{
	size_t vertex_count = mesh.streams[0].data.size();

	if (compressmore)
		meshopt_optimizeVertexCacheStrip(&mesh.indices[0], &mesh.indices[0], mesh.indices.size(), vertex_count);
	else
		meshopt_optimizeVertexCache(&mesh.indices[0], &mesh.indices[0], mesh.indices.size(), vertex_count);

	std::vector<unsigned int> remap(vertex_count);
	size_t unique_vertices = meshopt_optimizeVertexFetchRemap(&remap[0], &mesh.indices[0], mesh.indices.size(), vertex_count);
	assert(unique_vertices <= vertex_count);

	meshopt_remapIndexBuffer(&mesh.indices[0], &mesh.indices[0], mesh.indices.size(), &remap[0]);

	for (size_t i = 0; i < mesh.streams.size(); ++i)
	{
		assert(mesh.streams[i].data.size() == vertex_count);

		meshopt_remapVertexBuffer(&mesh.streams[i].data[0], &mesh.streams[i].data[0], vertex_count, sizeof(Attr), &remap[0]);
		mesh.streams[i].data.resize(unique_vertices);
	}
}

struct BoneInfluence
{
	float i;
	float w;
};

struct BoneInfluenceIndexPredicate
{
	bool operator()(const BoneInfluence& lhs, const BoneInfluence& rhs) const
	{
		return lhs.i < rhs.i;
	}
};

struct BoneInfluenceWeightPredicate
{
	bool operator()(const BoneInfluence& lhs, const BoneInfluence& rhs) const
	{
		return lhs.w > rhs.w;
	}
};

static void filterBones(Mesh& mesh)
{
	const int kMaxGroups = 8;

	std::pair<Stream*, Stream*> groups[kMaxGroups];
	int group_count = 0;

	// gather all joint/weight groups; each group contains 4 bone influences
	for (int i = 0; i < kMaxGroups; ++i)
	{
		Stream* jg = getStream(mesh, cgltf_attribute_type_joints, int(i));
		Stream* wg = getStream(mesh, cgltf_attribute_type_weights, int(i));

		if (!jg || !wg)
			break;

		groups[group_count++] = std::make_pair(jg, wg);
	}

	if (group_count == 0)
		return;

	// weights below cutoff can't be represented in quantized 8-bit storage
	const float weight_cutoff = 0.5f / 255.f;

	size_t vertex_count = mesh.streams[0].data.size();

	BoneInfluence inf[kMaxGroups * 4] = {};

	for (size_t i = 0; i < vertex_count; ++i)
	{
		int count = 0;

		// gather all bone influences for this vertex
		for (int j = 0; j < group_count; ++j)
		{
			const Attr& ja = groups[j].first->data[i];
			const Attr& wa = groups[j].second->data[i];

			for (int k = 0; k < 4; ++k)
				if (wa.f[k] > weight_cutoff)
				{
					inf[count].i = ja.f[k];
					inf[count].w = wa.f[k];
					count++;
				}
		}

		// pick top 4 influences - could use partial_sort but it is slower on small sets
		std::sort(inf, inf + count, BoneInfluenceWeightPredicate());

		// now sort top 4 influences by bone index - this improves compression ratio
		std::sort(inf, inf + std::min(4, count), BoneInfluenceIndexPredicate());

		// copy the top 4 influences back into stream 0 - we will remove other streams at the end
		Attr& ja = groups[0].first->data[i];
		Attr& wa = groups[0].second->data[i];

		for (int k = 0; k < 4; ++k)
		{
			if (k < count)
			{
				ja.f[k] = inf[k].i;
				wa.f[k] = inf[k].w;
			}
			else
			{
				ja.f[k] = 0.f;
				wa.f[k] = 0.f;
			}
		}
	}

	// remove redundant weight/joint streams
	for (size_t i = 0; i < mesh.streams.size();)
	{
		Stream& s = mesh.streams[i];

		if ((s.type == cgltf_attribute_type_joints || s.type == cgltf_attribute_type_weights) && s.index > 0)
			mesh.streams.erase(mesh.streams.begin() + i);
		else
			++i;
	}
}

static void simplifyPointMesh(Mesh& mesh, float threshold)
{
	if (threshold >= 1)
		return;

	const Stream* positions = getStream(mesh, cgltf_attribute_type_position);
	if (!positions)
		return;

	size_t vertex_count = mesh.streams[0].data.size();

	size_t target_vertex_count = size_t(double(vertex_count) * threshold);

	if (target_vertex_count < 1)
		return;

	std::vector<unsigned int> indices(target_vertex_count);
	indices.resize(meshopt_simplifyPoints(&indices[0], positions->data[0].f, vertex_count, sizeof(Attr), target_vertex_count));

	std::vector<Attr> scratch(indices.size());

	for (size_t i = 0; i < mesh.streams.size(); ++i)
	{
		std::vector<Attr>& data = mesh.streams[i].data;

		assert(data.size() == vertex_count);

		for (size_t j = 0; j < indices.size(); ++j)
			scratch[j] = data[indices[j]];

		data = scratch;
	}
}

static void sortPointMesh(Mesh& mesh)
{
	const Stream* positions = getStream(mesh, cgltf_attribute_type_position);
	if (!positions)
		return;

	size_t vertex_count = mesh.streams[0].data.size();

	std::vector<unsigned int> remap(vertex_count);
	meshopt_spatialSortRemap(&remap[0], positions->data[0].f, vertex_count, sizeof(Attr));

	for (size_t i = 0; i < mesh.streams.size(); ++i)
	{
		assert(mesh.streams[i].data.size() == vertex_count);

		meshopt_remapVertexBuffer(&mesh.streams[i].data[0], &mesh.streams[i].data[0], vertex_count, sizeof(Attr), &remap[0]);
	}
}

void processMesh(Mesh& mesh, const Settings& settings)
{
	filterStreams(mesh);

	switch (mesh.type)
	{
	case cgltf_primitive_type_points:
		assert(mesh.indices.empty());
		simplifyPointMesh(mesh, settings.simplify_threshold);
		sortPointMesh(mesh);
		break;

	case cgltf_primitive_type_triangles:
		filterBones(mesh);
		reindexMesh(mesh);
		filterTriangles(mesh);
		simplifyMesh(mesh, settings.simplify_threshold, settings.simplify_aggressive);
		optimizeMesh(mesh, settings.compressmore);
		break;

	default:
		assert(!"Unknown primitive type");
	}
}

bool getBoneRadius(std::vector<float>& result, const Mesh& mesh)
{
	if (!mesh.skin)
		return false;

	const Stream* positions = getStream(mesh, cgltf_attribute_type_position);
	const Stream* joints = getStream(mesh, cgltf_attribute_type_joints);
	const Stream* weights = getStream(mesh, cgltf_attribute_type_weights);

	if (!joints || !weights)
		return false;

	assert(joints->data.size() == positions->data.size() && weights->data.size() == positions->data.size());

	result.resize(mesh.skin->joints_count);

	std::vector<float> inverse_bind_matrices(mesh.skin->joints_count * 16);

	for (size_t j = 0; j < mesh.skin->joints_count; ++j)
	{
		float transform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

		if (mesh.skin->inverse_bind_matrices)
		{
			cgltf_accessor_read_float(mesh.skin->inverse_bind_matrices, j, transform, 16);
		}

		memcpy(&inverse_bind_matrices[j * 16], transform, sizeof(float) * 16);
	}

	for (size_t j = 0; j < joints->data.size(); ++j)
	{
		const Attr& vj = joints->data[j];
		const Attr& vw = weights->data[j];

		for (int k = 0; k < 4; ++k)
		{
			if (vw.f[k] == 0.f)
				continue;

			int bone = int(vj.f[k]);
			assert(size_t(bone) < result.size());

			Attr vp = positions->data[j];
			transformPosition(vp.f, &inverse_bind_matrices[bone * 16]);

			float vr = sqrtf(vp.f[0] * vp.f[0] + vp.f[1] * vp.f[1] + vp.f[2] * vp.f[2]);

			result[bone] = std::max(result[bone], vr);
		}
	}

	return true;
}
