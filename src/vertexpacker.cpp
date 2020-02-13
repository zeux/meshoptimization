// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <float.h>
#include <math.h> // fabsf

inline bool equal(float a, float b, float epsilon = 0.00001)
{
	return fabsf(a - b) < epsilon;
}

inline bool is_zero(float x, float epsilon = 0.00001)
{
	return fabsf(x) < epsilon;
}

inline float maxf(float a, float b)
{
	return a < b ? b : a;
}

inline float minf(float a, float b)
{
	return a > b ? b : a;
}

inline float clampf(float x, float min, float max)
{
	return minf(maxf(x, min), max);
}

// https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-data-conversion


inline int encodeUnorm(float v, int N)
{
    const float scale = float((1 << N) - 1);

    v = clampf(v, 0.0f, 1.0f);
    //v = (v >= 0) ? v : 0;
    //v = (v <= 1) ? v : 1;

	return int(v * scale + 0.5f);   // This is a truncating conversion (dropping decimal fraction).
}

inline float decodeUnorm(int v, int N)
{
	const float scale = float((1 << N) - 1);

	return float(v) / scale;
}

inline float pack(float f, float scale, float offset)
{
	return (f - offset) / scale;
}

inline float unpack(float f, float scale, float offset)
{
	return f * scale + offset;
}

// https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
inline float halfToFloat(unsigned short h)
{
	union FP32 {
		float f;
		unsigned int u;
	};

	static const FP32 magic = {(254 - 15) << 23};
	static const FP32 was_infnan = {(127 + 16) << 23};
	FP32 o;

	o.u = (h & 0x7fff) << 13; // exponent/mantissa bits
	o.f *= magic.f;           // exponent adjust
	if (o.f >= was_infnan.f)  // make sure Inf/NaN survive
		o.u |= 255 << 23;
	o.u |= (h & 0x8000) << 16; // sign bit
	return o.f;
}


// This function computes the best scale and offset that minimize the quantization error. It returns both the maximum error and the mean squared error.
// You would usually call this function with different quantization settings to determine which one to use, picking the smallest setting within the maximum allowed error.
// To minimize the error I assign each position to one quantization interval, then minimize the scale and offset under the assumption that the interval assignment
// doesn't change. This is usually not the case, so the process is repeated multiple times. This procedure tends to converge, but doesn't always do that. An 
// improvement would be to evaluate the error after each step and stop when it doesn't improve.
void meshopt_optimizeUnormQuantizationError(const float* vertex_ptr, size_t vertex_stride, size_t vertex_count, int bitcount, bool uniform_scale, int iteration_count, /*out*/ float scale[3], /*out*/ float offset[3])
{
	float minv[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float maxv[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	size_t vertex_stride_float = vertex_stride / sizeof(float);

	for (size_t i = 0; i < vertex_count; ++i)
	{
		const float* v = vertex_ptr + i * vertex_stride_float;

		for (int j = 0; j < 3; ++j)
		{
			minv[j] = minf(minv[j], v[j]);
			maxv[j] = maxf(maxv[j], v[j]);
		}
	}

	// Compute initial scale & offset based on bounding box.
	for (int j = 0; j < 3; ++j)
	{
		scale[j] = (maxv[j] - minv[j]);
		scale[j] = scale[j] < 0.01f ? 0.01f : scale[j]; // Avoid division by 0.
		offset[j] = minv[j];
	}

	bool best_offset_scale_changed = true;

    if (!uniform_scale)
	{
		// Optimize each dimension independently.
		for (int axis = 0; axis < 3; axis++)
		{
			float best_scale = scale[axis], best_offset = offset[axis];

			// Solving:

			// X = Q * m + a

			// In the least squares sense:

			// [XQ]=[QQ SQ][m]
			// [SX]=[SQ  n][a]

			// XQ = dot(X, Q)
			// QQ = dot(Q, Q)
			// SX = Sum(X,0,n-1)
			// SQ = Sum(Q,0,n-1)

			// Yields:

			// m = (XQ - SQ*SX/n) / (QQ - SQ*SQ/n)
			// a = (SX - SQ*m) / n

			// Refinement
			for (int it = 0; best_offset_scale_changed && it < iteration_count; it++)
			{
				best_offset_scale_changed = false;

				float XQ = 0;
				float SX = 0;
				float QQ = 0;
				float SQ = 0;

				// Compute squared matrix.
				for (int i = 0; i < vertex_count; i++)
				{
					float float_pos = vertex_ptr[i * vertex_stride_float + axis];

					// Encode inputs using initial offset-scale. Decode without applying scale/offset.
					float unorm_pos = decodeUnorm(encodeUnorm(pack(float_pos, best_scale, best_offset), bitcount), bitcount);

					// Add to least squares equations.
					XQ += float_pos * unorm_pos;
					SX += float_pos;
					QQ += unorm_pos * unorm_pos;
					SQ += unorm_pos;
				}

				// Solve least squares equations to obtain new offsets and scales.

				float n = float(vertex_count);

				const float epsilon = 0.0001f;
				float denom = QQ - SQ * SQ / n;
				float m = is_zero(denom) ? 1.0f : (XQ - SQ * SX / n) / denom;
				float a = (SX - SQ * m) / n;

				if (!equal(best_scale, m) || !equal(best_offset, a))
				{
					best_scale = m;
					best_offset = a;
					best_offset_scale_changed = true;
				}
			}

			scale[axis] = best_scale;
			offset[axis] = best_offset;
		}
	}
	else 
    {
		float best_scale = maxf(maxf(scale[0], scale[1]), scale[2]);
        float best_offset_x = offset[0];
		float best_offset_y = offset[1];
		float best_offset_z = offset[2];

		// Solving:

		// Xx = Qx * m + ax
		// Xy = Qy * m + ay
		// Xz = Qz * m + az

        // [Xx] = [Qx 1 0 0]
        // ...
	    // [Xy] = [Qy 0 1 0] [m ax ay az]^T
		// ...
		// [Xz] = [Qz 0 0 1]

		// In the least squares sense:

		// [XQ ]=[QQ  SQx SQy SQz][m ]
		// [SXx]=[SQx n   0   0  ][ax]
		// [SXy]=[SQy 0   n   0  ][ay]
		// [SXz]=[SQz 0   0   n  ][az]

		// XQ = dot(X, Q)
		// QQ = dot(Q, Q)
		// SX = Sum(X,0,n-1)
		// SQ = Sum(Q,0,n-1)

		// Yields:

		// m = (XQ - SQ*SX/n) / (QQ - SQ*SQ/n)
		// a = (SX - SQ*m) / n

    	for (int it = 0; best_offset_scale_changed && it < iteration_count; it++)
		{
			best_offset_scale_changed = false;

			float XQ = 0;
			float SXx = 0;
			float SXy = 0;
			float SXz = 0;
			float QQ = 0;
			float SQx = 0;
			float SQy = 0;
			float SQz = 0;

			// Compute squared matrix.
			for (int i = 0; i < vertex_count; i++)
			{
    		    float float_pos_x = vertex_ptr[i * vertex_stride_float + 0];
				float float_pos_y = vertex_ptr[i * vertex_stride_float + 1];
				float float_pos_z = vertex_ptr[i * vertex_stride_float + 2];

				// Encode inputs using initial offset-scale. Decode without applying scale/offset.
				float unorm_pos_x = decodeUnorm(encodeUnorm(pack(float_pos_x, best_scale, best_offset_x), bitcount), bitcount);
				float unorm_pos_y = decodeUnorm(encodeUnorm(pack(float_pos_y, best_scale, best_offset_y), bitcount), bitcount);
				float unorm_pos_z = decodeUnorm(encodeUnorm(pack(float_pos_z, best_scale, best_offset_z), bitcount), bitcount);

				// Add to least squares equations.
				XQ += float_pos_x * unorm_pos_x + float_pos_y * unorm_pos_y + float_pos_z * unorm_pos_z;
				SXx += float_pos_x;
				SXy += float_pos_y;
				SXz += float_pos_z;
				QQ += unorm_pos_x * unorm_pos_x + unorm_pos_y * unorm_pos_y + unorm_pos_z * unorm_pos_z;
				SQx += unorm_pos_x;
				SQy += unorm_pos_y;
				SQz += unorm_pos_z;
			}

            // Solve least squares equations to obtain new offsets and scale.

			float n = float(vertex_count);

            float denom = (QQ * n - SQz * SQz - SQy * SQy - SQx * SQx);
			const float epsilon = 0.0001f;
			if (fabsf(denom) < epsilon)
				break;

            float m = (XQ * n - SQx * SXx - SQy * SXy - SQz * SXz) / denom;
			//float ax = (XQ * (-SQx * n) + SQx * (QQ * n - SQz * SQz - SQy * SQy) + SQy * SQy * SQx + SQz * SQz * SQx) / denom;
			//float ay = (XQ * (-SQy * n) + SQx * SQx * SQy + SQy * (QQ * n - SQz * SQz - SQx * SQx) + SQz * SQz * SQy) / denom;
			//float az = (XQ * (-SQz * n) + SQx * SQx * SQz + SQy * SQy * SQz + SQz * (QQ * n - SQy * SQy - SQx * SQx)) / denom;

            float ax = (SXx - SQx * m) / n;
			float ay = (SXy - SQy * m) / n;
			float az = (SXz - SQz * m) / n;

			if (!equal(best_scale, m) || !equal(best_offset_x, ax) || !equal(best_offset_y, ay) || !equal(best_offset_z, az))
			{
				best_scale = m;
				best_offset_x = ax;
				best_offset_y = ay;
				best_offset_z = az;
				best_offset_scale_changed = true;
			}
		}

		scale[0] = scale[1] = scale[2] = best_scale;
		offset[0] = best_offset_x;
		offset[1] = best_offset_y;
		offset[2] = best_offset_z;
    }
}


void meshopt_evaluateUnormQuantizationError(const float* vertex_ptr, size_t vertex_stride, size_t vertex_count, int bitcount, float scale[3], float offset[3], /*out*/ float errors[2])
{
	double sq_error_sum = 0;
	float abs_error_max = 0;

	size_t vertex_stride_float = vertex_stride / sizeof(float);

	for (int i = 0; i < vertex_count; i++)
	{
		float fx = vertex_ptr[i * vertex_stride_float + 0];
		float fy = vertex_ptr[i * vertex_stride_float + 1];
		float fz = vertex_ptr[i * vertex_stride_float + 2];

		int bx = encodeUnorm(pack(fx, scale[0], offset[0]), bitcount);
		int by = encodeUnorm(pack(fy, scale[1], offset[1]), bitcount);
		int bz = encodeUnorm(pack(fz, scale[2], offset[2]), bitcount);

		float dfx = unpack(decodeUnorm(bx, bitcount), scale[0], offset[0]) - fx;
		float dfy = unpack(decodeUnorm(by, bitcount), scale[1], offset[1]) - fy;
		float dfz = unpack(decodeUnorm(bz, bitcount), scale[2], offset[2]) - fz;

		sq_error_sum += dfx * dfx;
		sq_error_sum += dfy * dfy;
		sq_error_sum += dfz * dfz;
		abs_error_max = maxf(abs_error_max, fabsf(dfx));
		abs_error_max = maxf(abs_error_max, fabsf(dfy));
		abs_error_max = maxf(abs_error_max, fabsf(dfz));
	}

	errors[0] = float(sq_error_sum / vertex_count);
	errors[1] = abs_error_max;
}


void meshopt_evaluateHalfQuantizationError(const float* vertex_ptr, size_t vertex_stride, size_t vertex_count, float scale[3], float offset[3], /*out*/ float errors[2])
{
	double sq_error_sum = 0;
	float abs_error_max = 0;

	size_t vertex_stride_float = vertex_stride / sizeof(float);

	for (int i = 0; i < vertex_count; i++)
	{
		float fx = vertex_ptr[i * vertex_stride_float + 0];
		float fy = vertex_ptr[i * vertex_stride_float + 1];
		float fz = vertex_ptr[i * vertex_stride_float + 2];

		auto hx = meshopt_quantizeHalf(pack(fx, scale[0], offset[0]));
		auto hy = meshopt_quantizeHalf(pack(fy, scale[1], offset[1]));
		auto hz = meshopt_quantizeHalf(pack(fz, scale[2], offset[2]));

		float dfx = unpack(halfToFloat(hx), scale[0], offset[0]) - fx;
		float dfy = unpack(halfToFloat(hy), scale[1], offset[1]) - fy;
		float dfz = unpack(halfToFloat(hz), scale[2], offset[2]) - fz;

		sq_error_sum += dfx * dfx;
		sq_error_sum += dfy * dfy;
		sq_error_sum += dfz * dfz;
		abs_error_max = maxf(abs_error_max, fabsf(dfx));
		abs_error_max = maxf(abs_error_max, fabsf(dfy));
		abs_error_max = maxf(abs_error_max, fabsf(dfz));
	}

	errors[0] = float(sq_error_sum / vertex_count);
	errors[1] = abs_error_max;
}


void meshopt_computeMeshBounds(const float* vertex_ptr, size_t vertex_stride, size_t vertex_count, float minv[3], float maxv[3])
{
	minv[0] = minv[1] = minv[2] = FLT_MAX;
	maxv[0] = maxv[1] = maxv[2] = -FLT_MAX;

	size_t vertex_stride_float = vertex_stride / sizeof(float);

	for (size_t i = 0; i < vertex_count; ++i)
	{
		const float* v = vertex_ptr + i * vertex_stride_float;

		for (int j = 0; j < 3; ++j)
		{
			minv[j] = minf(minv[j], v[j]);
			maxv[j] = maxf(maxv[j], v[j]);
		}
	}
}
