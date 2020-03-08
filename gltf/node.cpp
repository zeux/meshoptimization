// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <math.h>

#include <algorithm>

void markAnimated(cgltf_data* data, std::vector<NodeInfo>& nodes, const std::vector<Animation>& animations)
{
	for (size_t i = 0; i < animations.size(); ++i)
	{
		const Animation& animation = animations[i];

		for (size_t j = 0; j < animation.tracks.size(); ++j)
		{
			const Track& track = animation.tracks[j];

			// mark nodes that have animation tracks that change their base transform as animated
			if (!track.dummy)
			{
				NodeInfo& ni = nodes[track.node - data->nodes];

				ni.animated_paths |= (1 << track.path);
			}
		}
	}

	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		NodeInfo& ni = nodes[i];

		for (cgltf_node* node = &data->nodes[i]; node; node = node->parent)
			ni.animated |= nodes[node - data->nodes].animated_paths != 0;
	}
}

void markNeededNodes(cgltf_data* data, std::vector<NodeInfo>& nodes, const std::vector<Mesh>& meshes, const std::vector<Animation>& animations, const Settings& settings)
{
	// mark all joints as kept
	for (size_t i = 0; i < data->skins_count; ++i)
	{
		const cgltf_skin& skin = data->skins[i];

		// for now we keep all joints directly referenced by the skin and the entire ancestry tree; we keep names for joints as well
		for (size_t j = 0; j < skin.joints_count; ++j)
		{
			NodeInfo& ni = nodes[skin.joints[j] - data->nodes];

			ni.keep = true;
		}
	}

	// mark all animated nodes as kept
	for (size_t i = 0; i < animations.size(); ++i)
	{
		const Animation& animation = animations[i];

		for (size_t j = 0; j < animation.tracks.size(); ++j)
		{
			const Track& track = animation.tracks[j];

			if (settings.anim_const || !track.dummy)
			{
				NodeInfo& ni = nodes[track.node - data->nodes];

				ni.keep = true;
			}
		}
	}

	// mark all mesh nodes as kept
	for (size_t i = 0; i < meshes.size(); ++i)
	{
		const Mesh& mesh = meshes[i];

		if (mesh.node)
		{
			NodeInfo& ni = nodes[mesh.node - data->nodes];

			ni.keep = true;
		}
	}

	// mark all light/camera nodes as kept
	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		const cgltf_node& node = data->nodes[i];

		if (node.light || node.camera)
		{
			nodes[i].keep = true;
		}
	}

	// mark all named nodes as needed (if -kn is specified)
	if (settings.keep_named)
	{
		for (size_t i = 0; i < data->nodes_count; ++i)
		{
			const cgltf_node& node = data->nodes[i];

			if (node.name && *node.name)
			{
				nodes[i].keep = true;
			}
		}
	}
}

static float det3x3(const float* transform)
{
	float a0 = transform[5] * transform[10] - transform[6] * transform[9];
	float a1 = transform[4] * transform[10] - transform[6] * transform[8];
	float a2 = transform[4] * transform[9] - transform[5] * transform[8];

	return transform[0] * a0 - transform[1] * a1 + transform[2] * a2;
}

static void analyzeBone(cgltf_data* data, cgltf_node* node, std::vector<NodeInfo>& nodes, float scale)
{
	NodeInfo& ni = nodes[node - data->nodes];

	ni.radius_scale = scale;

	float transform[16];
	cgltf_node_transform_local(node, transform);

	float scale_next = scale * powf(fabsf(det3x3(transform)), 1.f / 3.f);

	ni.radius_tree = ni.radius_self;

	for (size_t i = 0; i < node->children_count; ++i)
	{
		cgltf_node* child = node->children[i];
		NodeInfo& cni = nodes[child - data->nodes];

		analyzeBone(data, child, nodes, scale_next);

		float child_transform[16];
		cgltf_node_transform_local(child, child_transform);

		float child_scale = powf(fabsf(det3x3(child_transform)), 1.f / 3.f);
		float child_radius = sqrtf(child_transform[12] * child_transform[12] + child_transform[13] * child_transform[13] + child_transform[14] * child_transform[14]);

		ni.radius_tree = std::max(ni.radius_tree, child_scale * cni.radius_tree + child_radius);
	}
}

void analyzeBoneRadius(cgltf_data* data, std::vector<NodeInfo>& nodes, const std::vector<Mesh>& meshes)
{
	// first, go through the meshes and compute bone space radius based on influences
	for (size_t i = 0; i < meshes.size(); ++i)
	{
		const Mesh& mesh = meshes[i];

		if (!mesh.skin)
			continue;

		std::vector<float> r;
		if (!getBoneRadius(r, mesh))
			continue;

		for (size_t j = 0; j < mesh.skin->joints_count; ++j)
		{
			cgltf_node* joint = mesh.skin->joints[j];
			NodeInfo& ni = nodes[joint - data->nodes];

			ni.radius_self = std::max(ni.radius_self, r[j]);
		}
	}

	// now, compute radius_scale and radius_tree hierarchically, starting from roots
	float max_radius = 0;

	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		cgltf_node* n = data->nodes + i;

		if (n->parent)
			continue;

		analyzeBone(data, n, nodes, 1.f);

		max_radius = std::max(max_radius, nodes[i].radius_tree);
	}

	// finally, assign # of bits based on radii, taking the max. radius as a reference point
	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		NodeInfo& ni = nodes[i];

		int bits = 10 + log2(1.f + ni.radius_tree * ni.radius_scale / max_radius * 20);

		ni.bits = std::min(bits, 16);
	}
}

void remapNodes(cgltf_data* data, std::vector<NodeInfo>& nodes, size_t& node_offset)
{
	// to keep a node, we currently need to keep the entire ancestry chain
	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		if (!nodes[i].keep)
			continue;

		for (cgltf_node* node = &data->nodes[i]; node; node = node->parent)
			nodes[node - data->nodes].keep = true;
	}

	// generate sequential indices for all nodes; they aren't sorted topologically
	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		NodeInfo& ni = nodes[i];

		if (ni.keep)
		{
			ni.remap = int(node_offset);

			node_offset++;
		}
	}
}
