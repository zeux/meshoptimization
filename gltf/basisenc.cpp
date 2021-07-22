#ifdef WITH_BASISU

#ifdef __clang__
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-value"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#ifdef _MSC_VER
#pragma warning(disable: 4702) // unreachable code
#endif

#define BASISU_NO_ITERATOR_DEBUG_LEVEL

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_IX86) || defined(_M_X64))
#define BASISU_SUPPORT_SSE 1
#endif

#if defined(__SSE4_1__)
#define BASISU_SUPPORT_SSE 1
#endif

#include "encoder/apg_bmp.c"
#include "encoder/basisu_astc_decomp.cpp"
#include "encoder/basisu_etc.cpp"
#include "encoder/basisu_resample_filters.cpp"
#include "encoder/basisu_backend.cpp"
#include "encoder/basisu_frontend.cpp"
#include "encoder/basisu_resampler.cpp"
#include "encoder/basisu_basis_file.cpp"
#include "encoder/basisu_bc7enc.cpp"
#include "encoder/basisu_global_selector_palette_helpers.cpp"
#include "encoder/basisu_ssim.cpp"
#include "encoder/basisu_gpu_texture.cpp"
#include "encoder/basisu_uastc_enc.cpp"
#include "encoder/basisu_comp.cpp"
#include "encoder/basisu_kernels_sse.cpp"
#include "encoder/jpgd.cpp"
#include "encoder/basisu_enc.cpp"
#include "encoder/basisu_pvrtc1_4.cpp"
#include "encoder/lodepng.cpp"
#include "transcoder/basisu_transcoder.cpp"

#undef CLAMP
#include "zstd/zstd.c"

bool encodeBasisInternal(const char* input, const char* output, bool yflip, bool normal_map, bool linear, bool uastc, int uastc_l, float uastc_q, int etc1s_l, int etc1s_q, int zstd_l, int width, int height)
{
	using namespace basisu;

	static int once = (basisu_encoder_init(), 0);

	basist::etc1_global_selector_codebook sel_codebook(basist::g_global_selector_cb_size, basist::g_global_selector_cb);

	uint32_t num_threads = std::thread::hardware_concurrency();
	if (num_threads < 1)
		num_threads = 1;

	job_pool jpool(num_threads);

	basis_compressor_params params;

	params.m_multithreading = num_threads > 1;
	params.m_pJob_pool = &jpool;

	if (uastc)
	{
		static const uint32_t s_level_flags[TOTAL_PACK_UASTC_LEVELS] = { cPackUASTCLevelFastest, cPackUASTCLevelFaster, cPackUASTCLevelDefault, cPackUASTCLevelSlower, cPackUASTCLevelVerySlow };

		params.m_uastc = true;

		params.m_pack_uastc_flags &= ~cPackUASTCLevelMask;
		params.m_pack_uastc_flags |= s_level_flags[uastc_l];

		params.m_rdo_uastc = uastc_q > 0;
		params.m_rdo_uastc_quality_scalar = uastc_q;
		params.m_rdo_uastc_dict_size = 1024;
	}
	else
	{
		params.m_compression_level = etc1s_l;
		params.m_quality_level = etc1s_q;
		params.m_max_endpoint_clusters = 0;
		params.m_max_selector_clusters = 0;

		params.m_no_selector_rdo = normal_map;
		params.m_no_endpoint_rdo = normal_map;
	}

	params.m_perceptual = !linear;

	params.m_mip_gen = true;
	params.m_mip_srgb = !linear;

	params.m_resample_width = width;
	params.m_resample_height = height;

	params.m_y_flip = yflip;

	params.m_create_ktx2_file = true;
	params.m_ktx2_srgb_transfer_func = !linear;

	if (zstd_l)
	{
		params.m_ktx2_uastc_supercompression = basist::KTX2_SS_ZSTANDARD;
		params.m_ktx2_zstd_supercompression_level = zstd_l;
	}

	params.m_read_source_images = true;
	params.m_write_output_basis_files = true;
	params.m_pSel_codebook = &sel_codebook;

	params.m_source_filenames.resize(1);
	params.m_source_filenames[0] = input;

	params.m_out_filename = output;

	params.m_status_output = false;
		
	basis_compressor c;

	if (!c.init(params))
		return false;

	return c.process() == basis_compressor::cECSuccess;
}

#endif
