// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <algorithm>

#include <math.h>
#include <string.h>

// TODO
#include "../src/meshoptimizer.h"
#include <stdio.h>

static float getDelta(const Attr& l, const Attr& r, cgltf_animation_path_type type)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		return std::max(std::max(fabsf(l.f[0] - r.f[0]), fabsf(l.f[1] - r.f[1])), fabsf(l.f[2] - r.f[2]));

	case cgltf_animation_path_type_rotation:
		return acosf(std::min(1.f, fabsf(l.f[0] * r.f[0] + l.f[1] * r.f[1] + l.f[2] * r.f[2] + l.f[3] * r.f[3])));

	case cgltf_animation_path_type_scale:
		return std::max(std::max(fabsf(l.f[0] / r.f[0] - 1), fabsf(l.f[1] / r.f[1] - 1)), fabsf(l.f[2] / r.f[2] - 1));

	case cgltf_animation_path_type_weights:
		return fabsf(l.f[0] - r.f[0]);

	default:
		assert(!"Uknown animation path");
		return 0;
	}
}

static float getDeltaTolerance(cgltf_animation_path_type type)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		return 0.001f; // linear

	case cgltf_animation_path_type_rotation:
		return 0.001f; // radians

	case cgltf_animation_path_type_scale:
		return 0.001f; // ratio

	case cgltf_animation_path_type_weights:
		return 0.001f; // linear

	default:
		assert(!"Uknown animation path");
		return 0;
	}
}

static Attr interpolateLinear(const Attr& l, const Attr& r, float t, cgltf_animation_path_type type)
{
	if (type == cgltf_animation_path_type_rotation)
	{
		// Approximating slerp, https://zeux.io/2015/07/23/approximating-slerp/
		// We also handle quaternion double-cover
		float ca = l.f[0] * r.f[0] + l.f[1] * r.f[1] + l.f[2] * r.f[2] + l.f[3] * r.f[3];

		float d = fabsf(ca);
		float A = 1.0904f + d * (-3.2452f + d * (3.55645f - d * 1.43519f));
		float B = 0.848013f + d * (-1.06021f + d * 0.215638f);
		float k = A * (t - 0.5f) * (t - 0.5f) + B;
		float ot = t + t * (t - 0.5f) * (t - 1) * k;

		float t0 = 1 - ot;
		float t1 = ca > 0 ? ot : -ot;

		Attr lerp = {{
		    l.f[0] * t0 + r.f[0] * t1,
		    l.f[1] * t0 + r.f[1] * t1,
		    l.f[2] * t0 + r.f[2] * t1,
		    l.f[3] * t0 + r.f[3] * t1,
		}};

		float len = sqrtf(lerp.f[0] * lerp.f[0] + lerp.f[1] * lerp.f[1] + lerp.f[2] * lerp.f[2] + lerp.f[3] * lerp.f[3]);

		if (len > 0.f)
		{
			lerp.f[0] /= len;
			lerp.f[1] /= len;
			lerp.f[2] /= len;
			lerp.f[3] /= len;
		}

		return lerp;
	}
	else
	{
		Attr lerp = {{
		    l.f[0] * (1 - t) + r.f[0] * t,
		    l.f[1] * (1 - t) + r.f[1] * t,
		    l.f[2] * (1 - t) + r.f[2] * t,
		    l.f[3] * (1 - t) + r.f[3] * t,
		}};

		return lerp;
	}
}

static Attr interpolateHermite(const Attr& v0, const Attr& t0, const Attr& v1, const Attr& t1, float t, float dt, cgltf_animation_path_type type)
{
	float s0 = 1 + t * t * (2 * t - 3);
	float s1 = t + t * t * (t - 2);
	float s2 = 1 - s0;
	float s3 = t * t * (t - 1);

	float ts1 = dt * s1;
	float ts3 = dt * s3;

	Attr lerp = {{
	    s0 * v0.f[0] + ts1 * t0.f[0] + s2 * v1.f[0] + ts3 * t1.f[0],
	    s0 * v0.f[1] + ts1 * t0.f[1] + s2 * v1.f[1] + ts3 * t1.f[1],
	    s0 * v0.f[2] + ts1 * t0.f[2] + s2 * v1.f[2] + ts3 * t1.f[2],
	    s0 * v0.f[3] + ts1 * t0.f[3] + s2 * v1.f[3] + ts3 * t1.f[3],
	}};

	if (type == cgltf_animation_path_type_rotation)
	{
		float len = sqrtf(lerp.f[0] * lerp.f[0] + lerp.f[1] * lerp.f[1] + lerp.f[2] * lerp.f[2] + lerp.f[3] * lerp.f[3]);

		if (len > 0.f)
		{
			lerp.f[0] /= len;
			lerp.f[1] /= len;
			lerp.f[2] /= len;
			lerp.f[3] /= len;
		}
	}

	return lerp;
}

static void resampleKeyframes(std::vector<Attr>& data, const std::vector<float>& input, const std::vector<Attr>& output, cgltf_animation_path_type type, cgltf_interpolation_type interpolation, size_t components, int frames, float mint, int freq)
{
	size_t cursor = 0;

	for (int i = 0; i < frames; ++i)
	{
		float time = mint + float(i) / freq;

		while (cursor + 1 < input.size())
		{
			float next_time = input[cursor + 1];

			if (next_time > time)
				break;

			cursor++;
		}

		if (cursor + 1 < input.size())
		{
			float cursor_time = input[cursor + 0];
			float next_time = input[cursor + 1];

			float range = next_time - cursor_time;
			float inv_range = (range == 0.f) ? 0.f : 1.f / (next_time - cursor_time);
			float t = std::max(0.f, std::min(1.f, (time - cursor_time) * inv_range));

			for (size_t j = 0; j < components; ++j)
			{
				switch (interpolation)
				{
				case cgltf_interpolation_type_linear:
				{
					const Attr& v0 = output[(cursor + 0) * components + j];
					const Attr& v1 = output[(cursor + 1) * components + j];
					data.push_back(interpolateLinear(v0, v1, t, type));
				}
				break;

				case cgltf_interpolation_type_step:
				{
					const Attr& v = output[cursor * components + j];
					data.push_back(v);
				}
				break;

				case cgltf_interpolation_type_cubic_spline:
				{
					const Attr& v0 = output[(cursor * 3 + 1) * components + j];
					const Attr& b0 = output[(cursor * 3 + 2) * components + j];
					const Attr& a1 = output[(cursor * 3 + 3) * components + j];
					const Attr& v1 = output[(cursor * 3 + 4) * components + j];
					data.push_back(interpolateHermite(v0, b0, v1, a1, t, range, type));
				}
				break;

				default:
					assert(!"Unknown interpolation type");
				}
			}
		}
		else
		{
			size_t offset = (interpolation == cgltf_interpolation_type_cubic_spline) ? cursor * 3 + 1 : cursor;

			for (size_t j = 0; j < components; ++j)
			{
				const Attr& v = output[offset * components + j];
				data.push_back(v);
			}
		}
	}
}

static bool isTrackEqual(const std::vector<Attr>& data, cgltf_animation_path_type type, int frames, const Attr* value, size_t components)
{
	assert(data.size() == frames * components);

	float tolerance = getDeltaTolerance(type);

	for (int i = 0; i < frames; ++i)
	{
		for (size_t j = 0; j < components; ++j)
		{
			float delta = getDelta(value[j], data[i * components + j], type);

			if (delta > tolerance)
				return false;
		}
	}

	return true;
}

static void getBaseTransform(Attr* result, size_t components, cgltf_animation_path_type type, cgltf_node* node)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		memcpy(result->f, node->translation, 3 * sizeof(float));
		break;

	case cgltf_animation_path_type_rotation:
		memcpy(result->f, node->rotation, 4 * sizeof(float));
		break;

	case cgltf_animation_path_type_scale:
		memcpy(result->f, node->scale, 3 * sizeof(float));
		break;

	case cgltf_animation_path_type_weights:
		if (node->weights_count)
		{
			assert(node->weights_count == components);
			memcpy(result->f, node->weights, components * sizeof(float));
		}
		else if (node->mesh && node->mesh->weights_count)
		{
			assert(node->mesh->weights_count == components);
			memcpy(result->f, node->mesh->weights, components * sizeof(float));
		}
		break;

	default:
		assert(!"Unknown animation path");
	}
}

void processAnimation(Animation& animation, const Settings& settings)
{
	float mint = 0, maxt = 0;

	for (size_t i = 0; i < animation.tracks.size(); ++i)
	{
		const Track& track = animation.tracks[i];
		assert(!track.time.empty());

		mint = std::min(mint, track.time.front());
		maxt = std::max(maxt, track.time.back());
	}

	// round the number of frames to nearest but favor the "up" direction
	// this means that at 10 Hz resampling, we will try to preserve the last frame <10ms
	// but if the last frame is <2ms we favor just removing this data
	int frames = 1 + int((maxt - mint) * settings.anim_freq + 0.8f);

	animation.start = mint;
	animation.frames = frames;

	std::vector<Attr> base;

	for (size_t i = 0; i < animation.tracks.size(); ++i)
	{
		Track& track = animation.tracks[i];

		std::vector<Attr> result;
		resampleKeyframes(result, track.time, track.data, track.path, track.interpolation, track.components, frames, mint, settings.anim_freq);

		track.time.clear();
		track.data.swap(result);

		if (isTrackEqual(track.data, track.path, frames, &track.data[0], track.components))
		{
			// track is constant (equal to first keyframe), we only need the first keyframe
			track.data.resize(track.components);

			// track.dummy is true iff track redundantly sets up the value to be equal to default node transform
			base.resize(track.components);
			getBaseTransform(&base[0], track.components, track.path, track.node);

			track.dummy = isTrackEqual(track.data, track.path, 1, &base[0], track.components);
		}
	}
}

static void roundtripTranslation(float r[3], const float f[3], const Settings& settings)
{
	int bits = settings.trn_bits - 9;

	r[0] = meshopt_quantizeFloat(f[0], bits);
	r[1] = meshopt_quantizeFloat(f[1], bits);
	r[2] = meshopt_quantizeFloat(f[2], bits);
}

static void encodeQuat(int16_t v[4], const float f[4], int bits)
{
	const float scaler = sqrtf(2.f);

	// establish maximum quaternion component
	int qc = 0;
	qc = fabsf(f[1]) > fabsf(f[qc]) ? 1 : qc;
	qc = fabsf(f[2]) > fabsf(f[qc]) ? 2 : qc;
	qc = fabsf(f[3]) > fabsf(f[qc]) ? 3 : qc;

	// we use double-cover properties to discard the sign
	float sign = f[qc] < 0.f ? -1.f : 1.f;

	// note: we always encode a cyclical swizzle to be able to recover the order via rotation
	v[0] = int16_t(meshopt_quantizeSnorm(f[(qc + 1) & 3] * scaler * sign, bits));
	v[1] = int16_t(meshopt_quantizeSnorm(f[(qc + 2) & 3] * scaler * sign, bits));
	v[2] = int16_t(meshopt_quantizeSnorm(f[(qc + 3) & 3] * scaler * sign, bits));
	v[3] = int16_t((meshopt_quantizeSnorm(1.f, bits) & ~0xff) | qc);
}

static void decodeQuat(float r[4], const int16_t v[4])
{
	const float scale = 1.f / sqrtf(2.f);

	static const int order[4][4] = {
	    {1, 2, 3, 0},
	    {2, 3, 0, 1},
	    {3, 0, 1, 2},
	    {0, 1, 2, 3},
	};

	// recover scale from the high byte of the component
	int sf = v[3] | 0xff;
	float ss = scale / float(sf);

	// convert x/y/z to [-1..1] (scaled...)
	float x = float(v[0]) * ss;
	float y = float(v[1]) * ss;
	float z = float(v[2]) * ss;

	// reconstruct w as a square root; we clamp to 0.f to avoid NaN due to precision errors
	float ww = 1.f - x * x - y * y - z * z;
	float w = sqrtf(ww >= 0.f ? ww : 0.f);

	// rounded signed float->int
	int xf = int(x * 32767.f + (x >= 0.f ? 0.5f : -0.5f));
	int yf = int(y * 32767.f + (y >= 0.f ? 0.5f : -0.5f));
	int zf = int(z * 32767.f + (z >= 0.f ? 0.5f : -0.5f));
	int wf = int(w * 32767.f + 0.5f);

	int qc = v[3] & 3;

	// output order is dictated by input index
	r[order[qc][0]] = short(xf) / 32767.f;
	r[order[qc][1]] = short(yf) / 32767.f;
	r[order[qc][2]] = short(zf) / 32767.f;
	r[order[qc][3]] = short(wf) / 32767.f;
}

static void roundtripRotation(float r[4], const float f[4], const Settings& settings, const NodeInfo& ni)
{
	if (settings.compressmore)
	{
		int16_t v[4];
		encodeQuat(v, f, ni.bits);
		decodeQuat(r, v);
	}
	else
	{
		r[0] = meshopt_quantizeSnorm(f[0], 16) / 32767.f;
		r[1] = meshopt_quantizeSnorm(f[1], 16) / 32767.f;
		r[2] = meshopt_quantizeSnorm(f[2], 16) / 32767.f;
		r[3] = meshopt_quantizeSnorm(f[3], 16) / 32767.f;
	}
}

static void roundtripScale(float r[3], const float f[3], const Settings& settings)
{
	int bits = settings.scl_bits - 9;

	r[0] = meshopt_quantizeFloat(f[0], bits);
	r[1] = meshopt_quantizeFloat(f[1], bits);
	r[2] = meshopt_quantizeFloat(f[2], bits);
}

static void q2m(float m[16], const float q[4])
{
	float qx = q[0];
	float qy = q[1];
	float qz = q[2];
	float qw = q[3];

	m[0] = (1 - 2 * qy*qy - 2 * qz*qz);
	m[1] = (2 * qx*qy + 2 * qz*qw);
	m[2] = (2 * qx*qz - 2 * qy*qw);
	m[3] = 0.f;

	m[4] = (2 * qx*qy - 2 * qz*qw);
	m[5] = (1 - 2 * qx*qx - 2 * qz*qz);
	m[6] = (2 * qy*qz + 2 * qx*qw);
	m[7] = 0.f;

	m[8] = (2 * qx*qz + 2 * qy*qw);
	m[9] = (2 * qy*qz - 2 * qx*qw);
	m[10] = (1 - 2 * qx*qx - 2 * qy*qy);
	m[11] = 0.f;

	m[12] = 0.f;
	m[13] = 0.f;
	m[14] = 0.f;
	m[15] = 1.f;
}

void analyzeAnimation(cgltf_data* data, const std::vector<NodeInfo>& nodes, const Animation& animation, const Settings& settings)
{
	(void)nodes;

	printf("Analyzing animation %s\n", animation.name ? animation.name : "?");

	std::vector<cgltf_node> stash(data->nodes_count);
	memcpy(&stash[0], data->nodes, sizeof(cgltf_node) * data->nodes_count);

	std::vector<float> errorest(data->nodes_count);
	std::vector<float> errors(data->nodes_count);
	std::vector<float> positions(data->nodes_count * 3);

	for (int i = 0; i < animation.frames; ++i)
	{
		// apply animation frame as is
		for (size_t j = 0; j < animation.tracks.size(); ++j)
		{
			const Track& t = animation.tracks[j];
			const Attr& a = t.dummy ? t.data[0] : t.data[i];

			switch (t.path)
			{
			case cgltf_animation_path_type_translation:
				t.node->translation[0] = a.f[0];
				t.node->translation[1] = a.f[1];
				t.node->translation[2] = a.f[2];
				break;

			case cgltf_animation_path_type_rotation:
				t.node->rotation[0] = a.f[0];
				t.node->rotation[1] = a.f[1];
				t.node->rotation[2] = a.f[2];
				t.node->rotation[3] = a.f[3];
				break;

			case cgltf_animation_path_type_scale:
				t.node->scale[0] = a.f[0];
				t.node->scale[1] = a.f[1];
				t.node->scale[2] = a.f[2];
				break;

			default:;
			}
		}

		// record world-space position of all nodes (doesn't account for skinning for leaves)
		for (size_t j = 0; j < data->nodes_count; ++j)
		{
			float transform[16];
			cgltf_node_transform_world(data->nodes + j, transform);

			positions[j * 3 + 0] = transform[12];
			positions[j * 3 + 1] = transform[13];
			positions[j * 3 + 2] = transform[14];
		}

		// apply animation frame with quantization
		for (size_t j = 0; j < animation.tracks.size(); ++j)
		{
			const Track& t = animation.tracks[j];
			const Attr& a = t.dummy ? t.data[0] : t.data[i];

			switch (t.path)
			{
			case cgltf_animation_path_type_translation:
				roundtripTranslation(t.node->translation, a.f, settings);
				break;

			case cgltf_animation_path_type_rotation:
				roundtripRotation(t.node->rotation, a.f, settings, nodes[t.node - data->nodes]);
				break;

			case cgltf_animation_path_type_scale:
				roundtripScale(t.node->scale, a.f, settings);
				break;

			default:;
			}
		}

		// record world-space position errors
		for (size_t j = 0; j < data->nodes_count; ++j)
		{
			float transform[16];
			cgltf_node_transform_world(data->nodes + j, transform);

			float dx = positions[j * 3 + 0] - transform[12];
			float dy = positions[j * 3 + 1] - transform[13];
			float dz = positions[j * 3 + 2] - transform[14];

			float e = sqrtf(dx * dx + dy * dy + dz * dz);

			errors[j] = std::max(errors[j], e);
		}

		// compute estimated errors just from rotation data
		for (size_t j = 0; j < animation.tracks.size(); ++j)
		{
			const Track& t = animation.tracks[j];
			const Attr& a = t.dummy ? t.data[0] : t.data[i];

			if (t.path == cgltf_animation_path_type_rotation)
			{
				Attr r;
				roundtripRotation(r.f, a.f, settings, nodes[t.node - data->nodes]);

				size_t ni = t.node - data->nodes;

				float R = nodes[ni].radius_tree * nodes[ni].radius_scale;

				if (0)
				{
					// d = cos(angle / 2)
					float d = fabsf(a.f[0] * r.f[0] + a.f[1] * r.f[1] + a.f[2] * r.f[2] + a.f[3] * r.f[3]);
					d /= sqrtf(r.f[0] * r.f[0] + r.f[1] * r.f[1] + r.f[2] * r.f[2] + r.f[3] * r.f[3]);

					// sin(angle / 2) * R * 2 is linear error
					float e = sqrtf(std::max(0.f, 1.f - d * d)) * R * 2;

					errorest[ni] = std::max(errorest[ni], e);
				}
				else
				{
					float m1[16], m2[16];
					q2m(m1, a.f);
					q2m(m2, r.f);

					for (int c = 0; c < 3; ++c)
					{
						float ex = m1[0 + c] * R - m2[0 + c] * R;
						float ey = m1[4 + c] * R - m2[4 + c] * R;
						float ez = m1[8 + c] * R - m2[8 + c] * R;

						float e = sqrtf(ex * ex + ey * ey + ez * ez);

						errorest[ni] = std::max(errorest[ni], e);
					}
				}
			}
		}
	}

	size_t maxj = 0;
	size_t maxest = 0;

	for (size_t j = 0; j < data->nodes_count; ++j)
	{
		cgltf_node* node = data->nodes + j;

		if (0)
		printf("Node %s: error %f (est %f), radius %f, tree %f\n",
			node->name ? node->name : "?",
			errors[j], errorest[j],
			nodes[j].radius_self * nodes[j].radius_scale,
			nodes[j].radius_tree * nodes[j].radius_scale);

		if (errors[j] > errors[maxj])
			maxj = j;
		if (errorest[j] > errorest[maxest])
			maxest = j;
	}

	cgltf_node* maxnode = data->nodes + maxj;
	cgltf_node* maxestnode = data->nodes + maxest;

	printf("Max error: node %s, error %f\n", maxnode->name ? maxnode->name : "?", errors[maxj]);
	printf("Max est error: node %s, error %f\n", maxestnode->name ? maxestnode->name : "?", errorest[maxest]);

	memcpy(data->nodes, &stash[0], sizeof(cgltf_node) * data->nodes_count);
}
