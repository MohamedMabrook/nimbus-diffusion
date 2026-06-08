// Nimbus Diffusion
// Copyright (C) 2026 Mohamed Mabrok
// GPL v3 - see LICENSE
//
// OFX diffusion plugin for DaVinci Resolve. Lens and Print stages,
// anamorphic stretch, chromatic bloom.
//
// PSF math based on spektrafilm by Andrea Volpato (GPLv3).
// I rewrote it in C++ with separable IIR blur instead of FFT,
// added the two-stage setup and the extra controls.

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>

#ifdef _WIN32
#  define EXPORT extern "C" __declspec(dllexport)
#else
#  define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxPixels.h"
#include "ofxMultiThread.h"

static OfxHost               *gHost        = nullptr;
static OfxImageEffectSuiteV1 *gEffectSuite = nullptr;
static OfxPropertySuiteV1    *gPropSuite   = nullptr;
static OfxParameterSuiteV1   *gParamSuite  = nullptr;
static OfxMemorySuiteV1      *gMemorySuite = nullptr;

static const char *kPluginId = "com.nimbusdiffusor.NimbusDiffusion";
static const int kMajorVer = 2, kMinorVer = 2;

// param IDs
#define kParamMix           "Mix"
#define kParamLensActive    "LensActive"
#define kParamLensFilter    "LensFilterType"
#define kParamLensStrength  "LensStrength"
#define kParamLensSize      "LensSize"
#define kParamLensWarmth    "LensWarmth"
#define kParamPrintActive   "PrintActive"
#define kParamPrintFilter   "PrintFilterType"
#define kParamPrintStrength "PrintStrength"
#define kParamPrintSize     "PrintSize"
#define kParamPrintWarmth   "PrintWarmth"
// advanced
#define kParamPixelSize      "SensorPixelUm"
#define kParamStretch        "Stretch"
#define kParamChroma         "ChromaShift"
#define kParamLensCore       "LensCoreIntensity"
#define kParamLensCoreSize   "LensCoreSize"
#define kParamLensHalo       "LensHaloIntensity"
#define kParamLensHaloSize   "LensHaloSize"
#define kParamLensBloom      "LensBloomIntensity"
#define kParamLensBloomSize  "LensBloomSize"
#define kParamPrintCore      "PrintCoreIntensity"
#define kParamPrintCoreSize  "PrintCoreSize"
#define kParamPrintHalo      "PrintHaloIntensity"
#define kParamPrintHaloSize  "PrintHaloSize"
#define kParamPrintBloom     "PrintBloomIntensity"
#define kParamPrintBloomSize "PrintBloomSize"

// filter presets, from spektrafilm's characterization
static const int MAX_SUB = 4;

struct GroupCfg { double lambda_um, spread; int n; double alpha; };

struct FamilyCfg {
    const char *id, *label;
    GroupCfg core, halo, bloom;
    double w_c, w_h, w_b, warmth_base, gain;
};

static const FamilyCfg kFamilies[4] = {
    { "glimmerglass",   "Glimmerglass",
      {  10.0,1.5,2,0.0},{  50.0,2.0,3,0.0},{ 260.0,2.5,4,3.2},
      0.60,0.30,0.10,0.00,0.65 },
    { "black_pro_mist", "Black Pro-Mist",
      {  16.0,1.5,2,0.0},{  95.0,2.0,3,0.0},{ 380.0,2.5,4,3.5},
      0.40,0.47,0.13,0.65,0.75 },
    { "pro_mist",       "Pro-Mist",
      {  14.0,1.5,2,0.0},{ 150.0,2.0,3,0.0},{ 650.0,2.5,4,2.9},
      0.28,0.42,0.30,0.40,1.05 },
    { "cinebloom",      "CineBloom",
      {  20.0,1.5,2,0.0},{ 200.0,2.0,3,0.0},{1000.0,2.5,4,2.5},
      0.22,0.30,0.48,0.85,1.00 },
};
static const int kNumFamilies = 4;

static const double kStrengthBreaks[5]    = {0.125,0.25,0.5,1.0,2.0};
static const double kStrengthFractions[5] = {0.10, 0.20,0.35,0.55,0.75};
static const double kWarmthAxis[3]        = {+1.30,+0.15,-1.45};

// simple piecewise linear interpolation
static double interp_lin(double x, const double *xs, const double *ys, int n) {
    if (x<=xs[0]) return ys[0]; if (x>=xs[n-1]) return ys[n-1];
    for (int i=0;i<n-1;++i)
        if (x>=xs[i]&&x<=xs[i+1])
            return ys[i]+(x-xs[i])/(xs[i+1]-xs[i])*(ys[i+1]-ys[i]);
    return ys[n-1];
}

// map user-facing strength (0.125..2.0 stops) to scattering fraction p_s
static double strength_to_ps(double s, int fi) {
    if (s<=0) return 0;
    double lb[5]; for (int i=0;i<5;++i) lb[i]=std::log2(kStrengthBreaks[i]);
    double base = interp_lin(std::log2(std::max(s,1e-9)),lb,kStrengthFractions,5);
    return std::max(0.0,std::min(0.99,base*kFamilies[fi].gain));
}

// spread a single PSF group into sub-components along a log scale
static void expand_group(const GroupCfg &g, bool is_bloom,
                          double *lams, double *wts) {
    int n=g.n;
    if (n<=1||g.spread<=1.0) {
        lams[0]=g.lambda_um; wts[0]=1.0;
        for (int k=1;k<n;++k){lams[k]=g.lambda_um;wts[k]=0.0;} return;
    }
    double lo=std::log(g.lambda_um/g.spread), hi=std::log(g.lambda_um*g.spread);
    for (int k=0;k<n;++k) lams[k]=std::exp(lo+(double)k/(n-1)*(hi-lo));
    if (is_bloom) {
        double ws=0; for (int k=0;k<n;++k){wts[k]=std::pow(lams[k],2.0-g.alpha);ws+=wts[k];}
        for (int k=0;k<n;++k) wts[k]/=ws;
    } else { for (int k=0;k<n;++k) wts[k]=1.0/n; }
}

// per-channel halo weights — warmth shifts energy toward R or B sub-components
static void halo_channel_weights(const double *wts, int n, double warmth,
                                  double out[3][MAX_SUB]) {
    warmth=std::max(-1.5,std::min(1.5,warmth));
    if (n<2){ for (int c=0;c<3;++c) out[c][0]=wts[0]; return; }
    double g[MAX_SUB],wsum=0,wg=0;
    for (int k=0;k<n;++k) g[k]=-1.0+2.0*(double)k/(n-1);
    for (int k=0;k<n;++k){wsum+=wts[k];wg+=wts[k]*g[k];}
    double gm=(wsum>0)?(wg/wsum):0; for (int k=0;k<n;++k) g[k]-=gm;
    for (int c=0;c<3;++c){
        double s=0;
        for (int k=0;k<n;++k){out[c][k]=std::max(0.0,wts[k]*(1.0+warmth*kWarmthAxis[c]*g[k]));s+=out[c][k];}
        double sc=(s>0)?(wsum/s):1.0; for (int k=0;k<n;++k) out[c][k]*=sc;
    }
}

// separable IIR exponential blur — forward+backward pass, energy-conserving
// alpha = exp(-1/lambda), per-channel lambda so chromatic bloom can vary R/G/B
using Buf = std::vector<float>;

static void exp_blur_general(Buf &buf, int w, int h,
                              const double lam_h[4], const double lam_v[4])
{
    float ah[4],omah[4],sh[4], av[4],omav[4],sv[4];
    for (int c=0;c<4;++c){
        ah[c]  =(lam_h[c]>=0.5)?(float)std::exp(-1.0/lam_h[c]):0.0f;
        omah[c]=1.0f-ah[c]; sh[c]=1.0f/(1.0f+ah[c]);
        av[c]  =(lam_v[c]>=0.5)?(float)std::exp(-1.0/lam_v[c]):0.0f;
        omav[c]=1.0f-av[c]; sv[c]=1.0f/(1.0f+av[c]);
    }
    int mwh=std::max(w,h);
    std::vector<float> fwd(mwh*4), bwd(mwh*4);

    for (int y=0;y<h;++y){
        float *row=buf.data()+(size_t)y*w*4;
        for (int c=0;c<4;++c) fwd[c]=row[c];
        for (int x=1;x<w;++x)
            for (int c=0;c<4;++c)
                fwd[x*4+c]=omah[c]*row[x*4+c]+ah[c]*fwd[(x-1)*4+c];
        for (int c=0;c<4;++c) bwd[(w-1)*4+c]=row[(w-1)*4+c];
        for (int x=w-2;x>=0;--x)
            for (int c=0;c<4;++c)
                bwd[x*4+c]=omah[c]*row[x*4+c]+ah[c]*bwd[(x+1)*4+c];
        for (int x=0;x<w;++x)
            for (int c=0;c<4;++c)
                row[x*4+c]=(fwd[x*4+c]+bwd[x*4+c]-omah[c]*row[x*4+c])*sh[c];
    }
    std::vector<float> col(h*4);
    for (int x=0;x<w;++x){
        for (int y=0;y<h;++y)
            for (int c=0;c<4;++c)
                col[y*4+c]=buf[(size_t)(y*w+x)*4+c];
        for (int c=0;c<4;++c) fwd[c]=col[c];
        for (int y=1;y<h;++y)
            for (int c=0;c<4;++c)
                fwd[y*4+c]=omav[c]*col[y*4+c]+av[c]*fwd[(y-1)*4+c];
        for (int c=0;c<4;++c) bwd[(h-1)*4+c]=col[(h-1)*4+c];
        for (int y=h-2;y>=0;--y)
            for (int c=0;c<4;++c)
                bwd[y*4+c]=omav[c]*col[y*4+c]+av[c]*bwd[(y+1)*4+c];
        for (int y=0;y<h;++y)
            for (int c=0;c<4;++c)
                buf[(size_t)(y*w+x)*4+c]=(fwd[y*4+c]+bwd[y*4+c]-omav[c]*col[y*4+c])*sv[c];
    }
}

static void exp_blur_inplace(Buf &buf, int w, int h, double lam_h, double lam_v) {
    double lh[4]={lam_h,lam_h,lam_h,lam_h};
    double lv[4]={lam_v,lam_v,lam_v,lam_v};
    exp_blur_general(buf,w,h,lh,lv);
}

static Buf read_image(void *data, int w, int h, int rowBytes, int nComp) {
    Buf buf((size_t)w*h*4,0.0f);
    for (int y=0;y<h;++y){
        const float *row=(const float*)((const char*)data+(size_t)y*rowBytes);
        for (int x=0;x<w;++x){
            const float *px=row+x*nComp; float *out=&buf[(size_t)(y*w+x)*4];
            out[0]=px[0];
            out[1]=nComp>=2?px[1]:0.0f;
            out[2]=nComp>=3?px[2]:0.0f;
            out[3]=nComp>=4?px[3]:1.0f;
        }
    }
    return buf;
}

static void write_image(const Buf &buf, void *data, int w, int h,
                         int rowBytes, int nComp) {
    for (int y=0;y<h;++y){
        float *row=(float*)((char*)data+(size_t)y*rowBytes);
        for (int x=0;x<w;++x){
            const float *in=&buf[(size_t)(y*w+x)*4]; float *px=row+x*nComp;
            px[0]=in[0];
            if(nComp>=2)px[1]=in[1];
            if(nComp>=3)px[2]=in[2];
            if(nComp>=4)px[3]=in[3];
        }
    }
}

// accumulate one diffusion stage into psf_acc
// adds p_s * K_stage * src — p_s is baked in so you can just sum both stages
//
// same formula as spektrafilm:
//   result = src + mix * (psf_acc - ps_total * src)
static double accumulate_stage(
    const Buf &src, int W, int H,
    int fi, double p_s,
    double size, double warmth, double pixelUm, double sq_str,
    double coreI, double coreSz,
    double haloI, double haloSz,
    double bloomI, double bloomSz,
    double chroma,
    Buf &psf_acc, Buf &tmp)
{
    if (p_s <= 0.0) return 0.0;

    const FamilyCfg &fam = kFamilies[fi];
    double wc = std::max(0.0, fam.w_c*coreI);
    double wh = std::max(0.0, fam.w_h*haloI);
    double wb = std::max(0.0, fam.w_b*bloomI);
    double wt = wc+wh+wb;
    if (wt <= 0.0) return 0.0;
    wc/=wt; wh/=wt; wb/=wt;

    double eff_warmth = fam.warmth_base + warmth;

    GroupCfg cg = {fam.core.lambda_um *coreSz, fam.core.spread,  fam.core.n,  fam.core.alpha};
    GroupCfg hg = {fam.halo.lambda_um *haloSz, fam.halo.spread,  fam.halo.n,  fam.halo.alpha};
    GroupCfg bg = {fam.bloom.lambda_um*bloomSz,fam.bloom.spread, fam.bloom.n, fam.bloom.alpha};

    double cl[MAX_SUB],cw[MAX_SUB], hl[MAX_SUB],hw[MAX_SUB], bl[MAX_SUB],bw[MAX_SUB];
    expand_group(cg, false, cl, cw);
    expand_group(hg, false, hl, hw);
    expand_group(bg, true,  bl, bw);

    double halo_ch[3][MAX_SUB];
    halo_channel_weights(hw, hg.n, eff_warmth, halo_ch);

    size_t np = (size_t)W*H;

    // core — achromatic, same blur for all channels
    for (int k=0; k<cg.n; ++k) {
        if (cw[k]<=0) continue;
        double lam = cl[k]*size/pixelUm;
        exp_blur_inplace(tmp=src, W, H, lam*sq_str, lam/sq_str);
        float coeff = (float)(p_s*wc*cw[k]);
        for (size_t i=0; i<np; ++i) {
            psf_acc[i*4+0]+=coeff*tmp[i*4+0];
            psf_acc[i*4+1]+=coeff*tmp[i*4+1];
            psf_acc[i*4+2]+=coeff*tmp[i*4+2];
        }
    }

    // halo — per-channel weights for the warmth tint
    for (int k=0; k<hg.n; ++k) {
        double lam = hl[k]*size/pixelUm;
        exp_blur_inplace(tmp=src, W, H, lam*sq_str, lam/sq_str);
        float whr=(float)(p_s*wh*halo_ch[0][k]);
        float whg=(float)(p_s*wh*halo_ch[1][k]);
        float whb=(float)(p_s*wh*halo_ch[2][k]);
        for (size_t i=0; i<np; ++i) {
            psf_acc[i*4+0]+=whr*tmp[i*4+0];
            psf_acc[i*4+1]+=whg*tmp[i*4+1];
            psf_acc[i*4+2]+=whb*tmp[i*4+2];
        }
    }

    // bloom — chromatic aberration: R scatters wider, B tighter
    const double cs[4]={1.0+chroma, 1.0, std::max(0.1,1.0-chroma*0.5), 1.0};
    for (int k=0; k<bg.n; ++k) {
        if (bw[k]<=0) continue;
        double lam_base = bl[k]*size/pixelUm;
        float coeff = (float)(p_s*wb*bw[k]);
        if (chroma<1e-4) {
            exp_blur_inplace(tmp=src, W, H, lam_base*sq_str, lam_base/sq_str);
            for (size_t i=0; i<np; ++i) {
                psf_acc[i*4+0]+=coeff*tmp[i*4+0];
                psf_acc[i*4+1]+=coeff*tmp[i*4+1];
                psf_acc[i*4+2]+=coeff*tmp[i*4+2];
            }
        } else {
            double lh[4],lv[4];
            for (int c=0;c<4;++c){ lh[c]=lam_base*cs[c]*sq_str; lv[c]=lam_base*cs[c]/sq_str; }
            exp_blur_general(tmp=src, W, H, lh, lv);
            for (size_t i=0; i<np; ++i) {
                psf_acc[i*4+0]+=coeff*tmp[i*4+0];
                psf_acc[i*4+1]+=coeff*tmp[i*4+1];
                psf_acc[i*4+2]+=coeff*tmp[i*4+2];
            }
        }
    }
    return p_s;
}

static OfxStatus action_load() {
    gEffectSuite=(OfxImageEffectSuiteV1*)gHost->fetchSuite(gHost->host,kOfxImageEffectSuite,1);
    gPropSuite  =(OfxPropertySuiteV1*)  gHost->fetchSuite(gHost->host,kOfxPropertySuite,   1);
    gParamSuite =(OfxParameterSuiteV1*) gHost->fetchSuite(gHost->host,kOfxParameterSuite,  1);
    gMemorySuite=(OfxMemorySuiteV1*)    gHost->fetchSuite(gHost->host,kOfxMemorySuite,      1);
    if(!gEffectSuite||!gPropSuite||!gParamSuite) return kOfxStatErrMissingHostFeature;
    return kOfxStatOK;
}

static OfxStatus action_describe(OfxImageEffectHandle handle) {
    OfxPropertySetHandle props;
    gEffectSuite->getPropertySet(handle,&props);
    gPropSuite->propSetString(props,kOfxPropLabel,                           0,"Nimbus Diffusion");
    gPropSuite->propSetString(props,kOfxImageEffectPluginPropGrouping,       0,"Film");
    gPropSuite->propSetString(props,kOfxPropShortLabel,                      0,"Nimbus Diffusion");
    gPropSuite->propSetString(props,kOfxPropLongLabel,                       0,"Nimbus Diffusion");
    gPropSuite->propSetString(props,kOfxPropIcon,                            0,"NimbusDiffusion");
    gPropSuite->propSetString(props,kOfxImageEffectPropSupportedPixelDepths, 0,kOfxBitDepthFloat);
    gPropSuite->propSetString(props,kOfxImageEffectPropSupportedContexts,    0,kOfxImageEffectContextFilter);
    gPropSuite->propSetInt   (props,kOfxImageEffectPropSupportsMultipleClipDepths,0,0);
    return kOfxStatOK;
}

// helpers for declaring params without repeating 8 lines every time
#define DEF_D(id_,lbl_,hint_,def_,mn_,mx_,dmn_,dmx_,par_)                       \
    gParamSuite->paramDefine(paramSet,kOfxParamTypeDouble,id_,&pProps);           \
    gPropSuite->propSetString(pProps,kOfxPropLabel,           0,lbl_);            \
    gPropSuite->propSetString(pProps,kOfxParamPropHint,       0,hint_);           \
    gPropSuite->propSetDouble(pProps,kOfxParamPropDefault,    0,def_);            \
    gPropSuite->propSetDouble(pProps,kOfxParamPropMin,        0,mn_);             \
    gPropSuite->propSetDouble(pProps,kOfxParamPropMax,        0,mx_);             \
    gPropSuite->propSetDouble(pProps,kOfxParamPropDisplayMin, 0,dmn_);            \
    gPropSuite->propSetDouble(pProps,kOfxParamPropDisplayMax, 0,dmx_);            \
    gPropSuite->propSetString(pProps,kOfxParamPropDoubleType, 0,kOfxParamDoubleTypePlain); \
    if((par_)[0]) gPropSuite->propSetString(pProps,kOfxParamPropParent,0,par_);

#define DEF_B(id_,lbl_,hint_,def_,par_)                                          \
    gParamSuite->paramDefine(paramSet,kOfxParamTypeBoolean,id_,&pProps);          \
    gPropSuite->propSetString(pProps,kOfxPropLabel,         0,lbl_);              \
    gPropSuite->propSetString(pProps,kOfxParamPropHint,     0,hint_);             \
    gPropSuite->propSetInt   (pProps,kOfxParamPropDefault,  0,(def_)?1:0);        \
    if((par_)[0]) gPropSuite->propSetString(pProps,kOfxParamPropParent,0,par_);

// 5 main controls per stage
#define DEF_STAGE_BASIC(ACTIVE_,FILT_,STR_,SZ_,WM_, ACTIVE_DEF_,STR_DEF_, GRP_) \
    DEF_B(ACTIVE_,"Active","Turn this stage on or off.",ACTIVE_DEF_,GRP_)         \
    gParamSuite->paramDefine(paramSet,kOfxParamTypeChoice,FILT_,&pProps);         \
    gPropSuite->propSetString(pProps,kOfxPropLabel,            0,"Filter Type");  \
    gPropSuite->propSetString(pProps,kOfxParamPropHint,        0,"Which filter to emulate."); \
    gPropSuite->propSetInt   (pProps,kOfxParamPropDefault,     0,1);              \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,0,"Glimmerglass"); \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,1,"Black Pro-Mist"); \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,2,"Pro-Mist");     \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,3,"CineBloom");    \
    gPropSuite->propSetString(pProps,kOfxParamPropParent,      0,GRP_);           \
    DEF_D(STR_,"Strength",                                                        \
        "How strong the filter is. Matches commercial stop ratings: 0.125=1/8, 0.25=1/4, 0.5=1/2, 1.0=full, 2.0=heavy.", \
        STR_DEF_, 0.0,2.0, 0.0,2.0, GRP_)                                        \
    DEF_D(SZ_,"Size","Scale all blur radii up or down.",                          \
        1.0, 0.1,4.0, 0.1,4.0, GRP_)                                             \
    DEF_D(WM_,"Warmth","Tint the halo. +1 = warm yellowish glow, -1 = cool blue.", \
        0.0, -1.5,1.5, -1.5,1.5, GRP_)

// 6 fine-tuning controls per stage (under Advanced)
#define DEF_STAGE_ADV(CI_,CS_,HI_,HS_,BI_,BS_, GRP_)                            \
    DEF_D(CI_,"Core Intensity","Core brightness multiplier.",                     \
        1.0, 0.0,4.0, 0.0,4.0, GRP_)                                             \
    DEF_D(CS_,"Core Size","Core radius multiplier.",                              \
        1.0, 0.1,4.0, 0.1,4.0, GRP_)                                             \
    DEF_D(HI_,"Halo Intensity","Halo brightness multiplier.",                     \
        1.0, 0.0,4.0, 0.0,4.0, GRP_)                                             \
    DEF_D(HS_,"Halo Size","Halo radius multiplier.",                              \
        1.0, 0.1,4.0, 0.1,4.0, GRP_)                                             \
    DEF_D(BI_,"Bloom Intensity","Bloom brightness multiplier.",                   \
        1.0, 0.0,4.0, 0.0,4.0, GRP_)                                             \
    DEF_D(BS_,"Bloom Size","Bloom radius multiplier.",                            \
        1.0, 0.1,4.0, 0.1,4.0, GRP_)

static OfxStatus action_describe_in_context(OfxImageEffectHandle handle,
                                             OfxPropertySetHandle /*inArgs*/)
{
    OfxPropertySetHandle clipProps;
    gEffectSuite->clipDefine(handle,kOfxImageEffectSimpleSourceClipName,&clipProps);
    gPropSuite->propSetString(clipProps,kOfxImageEffectPropSupportedComponents,0,kOfxImageComponentRGBA);
    gPropSuite->propSetString(clipProps,kOfxImageEffectPropSupportedComponents,1,kOfxImageComponentRGB);
    gEffectSuite->clipDefine(handle,kOfxImageEffectOutputClipName,&clipProps);
    gPropSuite->propSetString(clipProps,kOfxImageEffectPropSupportedComponents,0,kOfxImageComponentRGBA);
    gPropSuite->propSetString(clipProps,kOfxImageEffectPropSupportedComponents,1,kOfxImageComponentRGB);

    OfxParamSetHandle paramSet;
    gEffectSuite->getParamSet(handle,&paramSet);
    OfxPropertySetHandle pProps;

    DEF_D(kParamMix, "Mix",
        "Blend between original and diffused. 0=off, 1=full.",
        1.0, 0.0,1.0, 0.0,1.0, "")

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"LensGroup",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,         0,"Lens");
    gPropSuite->propSetInt   (pProps,kOfxParamPropGroupOpen,0,1);

    DEF_STAGE_BASIC(
        kParamLensActive, kParamLensFilter, kParamLensStrength,
        kParamLensSize,   kParamLensWarmth,
        /*active_def=*/true, /*str_def=*/1.0, "LensGroup")

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"LensGroupEnd",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,       0,"");
    gPropSuite->propSetInt   (pProps,kOfxParamPropSecret, 0,1);

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"PrintGroup",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,         0,"Print");
    gPropSuite->propSetInt   (pProps,kOfxParamPropGroupOpen,0,0);

    DEF_STAGE_BASIC(
        kParamPrintActive, kParamPrintFilter, kParamPrintStrength,
        kParamPrintSize,   kParamPrintWarmth,
        /*active_def=*/false, /*str_def=*/0.5, "PrintGroup")

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"PrintGroupEnd",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,       0,"");
    gPropSuite->propSetInt   (pProps,kOfxParamPropSecret, 0,1);

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"AdvGroup",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,         0,"Advanced");
    gPropSuite->propSetInt   (pProps,kOfxParamPropGroupOpen,0,0);

    DEF_D(kParamPixelSize,"Sensor Pixel um",
        "Physical pixel size in micrometers. 6.0 is typical for a cinema camera. "
        "Affects the absolute scale of the blur.",
        6.0, 1.0,24.0, 1.0,24.0, "AdvGroup")

    DEF_D(kParamStretch,"Anamorphic Stretch",
        "Stretch the glow horizontally. 1.0=round, 2.0=oval twice as wide as tall.",
        1.0, 0.5,4.0, 0.5,4.0, "AdvGroup")

    DEF_D(kParamChroma,"Chroma Shift",
        "Chromatic aberration in the bloom. Red scatters further, blue less. 0=off.",
        0.0, 0.0,0.5, 0.0,0.5, "AdvGroup")

    DEF_STAGE_ADV(
        kParamLensCore,  kParamLensCoreSize,
        kParamLensHalo,  kParamLensHaloSize,
        kParamLensBloom, kParamLensBloomSize,
        "AdvGroup")

    DEF_STAGE_ADV(
        kParamPrintCore,  kParamPrintCoreSize,
        kParamPrintHalo,  kParamPrintHaloSize,
        kParamPrintBloom, kParamPrintBloomSize,
        "AdvGroup")

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"AdvGroupEnd",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,       0,"");
    gPropSuite->propSetInt   (pProps,kOfxParamPropSecret, 0,1);

    return kOfxStatOK;
}

static OfxStatus action_is_identity(OfxImageEffectHandle handle,
                                     OfxPropertySetHandle inArgs,
                                     OfxPropertySetHandle outArgs) {
    double time; gPropSuite->propGetDouble(inArgs,kOfxPropTime,0,&time);
    OfxParamSetHandle ps; gEffectSuite->getParamSet(handle,&ps);

    auto gd=[&](const char *id,double def)->double{
        OfxParamHandle ph; gParamSuite->paramGetHandle(ps,id,&ph,nullptr);
        double v=def; gParamSuite->paramGetValueAtTime(ph,time,&v); return v;
    };
    auto gi=[&](const char *id,int def)->int{
        OfxParamHandle ph; gParamSuite->paramGetHandle(ps,id,&ph,nullptr);
        int v=def; gParamSuite->paramGetValueAtTime(ph,time,&v); return v;
    };

    if (gd(kParamMix,1.0) <= 0) {
        gPropSuite->propSetString(outArgs,kOfxPropName,0,kOfxImageEffectSimpleSourceClipName);
        gPropSuite->propSetDouble(outArgs,kOfxPropTime,0,time);
        return kOfxStatOK;
    }
    bool lensOn  = gi(kParamLensActive, 1) && gd(kParamLensStrength, 1.0)>0;
    bool printOn = gi(kParamPrintActive,0) && gd(kParamPrintStrength,0.5)>0;
    if (!lensOn && !printOn) {
        gPropSuite->propSetString(outArgs,kOfxPropName,0,kOfxImageEffectSimpleSourceClipName);
        gPropSuite->propSetDouble(outArgs,kOfxPropTime,0,time);
        return kOfxStatOK;
    }
    return kOfxStatReplyDefault;
}

static OfxStatus action_get_roi(OfxImageEffectHandle handle,
                                 OfxPropertySetHandle inArgs,
                                 OfxPropertySetHandle outArgs) {
    double time; gPropSuite->propGetDouble(inArgs,kOfxPropTime,0,&time);
    OfxParamSetHandle ps; gEffectSuite->getParamSet(handle,&ps);

    auto fd=[&](const char *id,double def)->double{
        OfxParamHandle ph; gParamSuite->paramGetHandle(ps,id,&ph,nullptr);
        double v=def; gParamSuite->paramGetValueAtTime(ph,time,&v); return v;
    };
    auto fi_of=[&](const char *id)->int{
        OfxParamHandle ph; gParamSuite->paramGetHandle(ps,id,&ph,nullptr);
        int v=1; gParamSuite->paramGetValueAtTime(ph,time,&v);
        return std::max(0,std::min(kNumFamilies-1,v));
    };

    double pixelUm = std::max(1.0, fd(kParamPixelSize,6.0));
    double stretch = std::max(0.5, fd(kParamStretch,  1.0));
    double chroma  = std::max(0.0, fd(kParamChroma,   0.0));

    auto stage_lam=[&](const char *filt, const char *sz, const char *bsz)->double{
        int fi=fi_of(filt);
        double size=std::max(0.1,fd(sz,1.0)), bsz_v=std::max(0.1,fd(bsz,1.0));
        return kFamilies[fi].bloom.lambda_um*kFamilies[fi].bloom.spread
               *bsz_v*size*(1.0+chroma)/pixelUm;
    };

    double lam_max=std::max(
        stage_lam(kParamLensFilter, kParamLensSize, kParamLensBloomSize),
        stage_lam(kParamPrintFilter,kParamPrintSize,kParamPrintBloomSize));

    double sq_str=std::sqrt(stretch);
    double border=std::ceil(7.0*lam_max*std::max(sq_str,1.0/sq_str));
    border=std::min(border,8192.0);

    double roi[4];
    gPropSuite->propGetDoubleN(inArgs,kOfxImageEffectPropRegionOfInterest,4,roi);
    double src[4]={roi[0]-border,roi[1]-border,roi[2]+border,roi[3]+border};
    std::string prop=std::string("OfxImageClipPropRoI_")+kOfxImageEffectSimpleSourceClipName;
    gPropSuite->propSetDoubleN(outArgs,prop.c_str(),4,src);
    return kOfxStatOK;
}

static OfxStatus action_render(OfxImageEffectHandle handle,
                                OfxPropertySetHandle inArgs)
{
    double time; gPropSuite->propGetDouble(inArgs,kOfxPropTime,0,&time);
    OfxParamSetHandle ps; gEffectSuite->getParamSet(handle,&ps);

    auto gd=[&](const char *id,double def)->double{
        OfxParamHandle ph; gParamSuite->paramGetHandle(ps,id,&ph,nullptr);
        double v=def; gParamSuite->paramGetValueAtTime(ph,time,&v); return v;
    };
    auto gi=[&](const char *id,int def)->int{
        OfxParamHandle ph; gParamSuite->paramGetHandle(ps,id,&ph,nullptr);
        int v=def; gParamSuite->paramGetValueAtTime(ph,time,&v); return v;
    };

    double mix    = std::max(0.0,std::min(1.0, gd(kParamMix,      1.0)));
    double pixelUm= std::max(1e-6,              gd(kParamPixelSize,6.0));
    double stretch= std::max(0.01,              gd(kParamStretch,  1.0));
    double chroma = std::max(0.0,               gd(kParamChroma,   0.0));
    double sq_str = std::sqrt(stretch);

    bool   lensOn = (bool)gi(kParamLensActive, 1);
    int    lensFi = std::max(0,std::min(kNumFamilies-1,gi(kParamLensFilter,1)));
    double lensSt = gd(kParamLensStrength,1.0);
    double lensSz = std::max(1e-6, gd(kParamLensSize,   1.0));
    double lensWm =                gd(kParamLensWarmth, 0.0);
    double lensCI = std::max(0.0,  gd(kParamLensCore,   1.0));
    double lensCS = std::max(1e-6, gd(kParamLensCoreSize,1.0));
    double lensHI = std::max(0.0,  gd(kParamLensHalo,   1.0));
    double lensHS = std::max(1e-6, gd(kParamLensHaloSize,1.0));
    double lensBI = std::max(0.0,  gd(kParamLensBloom,  1.0));
    double lensBS = std::max(1e-6, gd(kParamLensBloomSize,1.0));
    double lensPS = (lensOn && lensSt>0) ? strength_to_ps(lensSt,lensFi) : 0.0;

    bool   prnOn = (bool)gi(kParamPrintActive, 0);
    int    prnFi = std::max(0,std::min(kNumFamilies-1,gi(kParamPrintFilter,1)));
    double prnSt = gd(kParamPrintStrength,0.5);
    double prnSz = std::max(1e-6, gd(kParamPrintSize,   1.0));
    double prnWm =                gd(kParamPrintWarmth, 0.0);
    double prnCI = std::max(0.0,  gd(kParamPrintCore,   1.0));
    double prnCS = std::max(1e-6, gd(kParamPrintCoreSize,1.0));
    double prnHI = std::max(0.0,  gd(kParamPrintHalo,   1.0));
    double prnHS = std::max(1e-6, gd(kParamPrintHaloSize,1.0));
    double prnBI = std::max(0.0,  gd(kParamPrintBloom,  1.0));
    double prnBS = std::max(1e-6, gd(kParamPrintBloomSize,1.0));
    double prnPS = (prnOn && prnSt>0) ? strength_to_ps(prnSt,prnFi) : 0.0;

    OfxImageClipHandle srcClip,dstClip;
    gEffectSuite->clipGetHandle(handle,kOfxImageEffectSimpleSourceClipName,&srcClip,nullptr);
    gEffectSuite->clipGetHandle(handle,kOfxImageEffectOutputClipName,&dstClip,nullptr);
    OfxPropertySetHandle srcImg,dstImg;
    gEffectSuite->clipGetImage(srcClip,time,nullptr,&srcImg);
    gEffectSuite->clipGetImage(dstClip,time,nullptr,&dstImg);
    auto release=[&](){gEffectSuite->clipReleaseImage(srcImg);gEffectSuite->clipReleaseImage(dstImg);};

    char *srcDepth=nullptr;
    gPropSuite->propGetString(srcImg,kOfxImageEffectPropPixelDepth,0,&srcDepth);
    if(!srcDepth||strcmp(srcDepth,kOfxBitDepthFloat)!=0){release();return kOfxStatFailed;}

    int sb[4]={},db[4]={};
    gPropSuite->propGetIntN(srcImg,kOfxImagePropBounds,4,sb);
    gPropSuite->propGetIntN(dstImg,kOfxImagePropBounds,4,db);
    int srb,drb; void *sd,*dd;
    gPropSuite->propGetInt    (srcImg,kOfxImagePropRowBytes,0,&srb);
    gPropSuite->propGetInt    (dstImg,kOfxImagePropRowBytes,0,&drb);
    gPropSuite->propGetPointer(srcImg,kOfxImagePropData,   0,&sd);
    gPropSuite->propGetPointer(dstImg,kOfxImagePropData,   0,&dd);
    char *sc=nullptr,*dc=nullptr;
    gPropSuite->propGetString(srcImg,kOfxImageEffectPropComponents,0,&sc);
    gPropSuite->propGetString(dstImg,kOfxImageEffectPropComponents,0,&dc);
    int snc=(sc&&!strcmp(sc,kOfxImageComponentRGB))?3:4;
    int dnc=(dc&&!strcmp(dc,kOfxImageComponentRGB))?3:4;

    int srcW=sb[2]-sb[0], srcH=sb[3]-sb[1];
    int dstW=db[2]-db[0], dstH=db[3]-db[1];

    // pass source through if there's nothing to do — avoids a white frame at Strength=0
    bool hasEffect = (mix>0) && (lensPS>0 || prnPS>0);
    if (!hasEffect || srcW<=0||srcH<=0||dstW<=0||dstH<=0) {
        if (srcW>0&&srcH>0&&dstW>0&&dstH>0) {
            Buf pt=read_image(sd,srcW,srcH,srb,snc);
            int ox2=db[0]-sb[0], oy2=db[1]-sb[1];
            Buf out((size_t)dstW*dstH*4,0.0f);
            for (int y=0;y<dstH;++y){ int sy=y+oy2; if(sy<0||sy>=srcH) continue;
                for (int x=0;x<dstW;++x){ int sx=x+ox2; if(sx<0||sx>=srcW) continue;
                    size_t si=(size_t)(sy*srcW+sx)*4, di=(size_t)(y*dstW+x)*4;
                    out[di+0]=pt[si+0]; out[di+1]=pt[si+1];
                    out[di+2]=pt[si+2]; out[di+3]=pt[si+3];
                }
            }
            write_image(out,dd,dstW,dstH,drb,dnc);
        }
        release(); return kOfxStatOK;
    }

    Buf src = read_image(sd,srcW,srcH,srb,snc);

    // sum the two stages into psf_acc, then combine:
    //   result = src + mix * (psf_acc - ps_total * src)
    Buf psf_acc((size_t)srcW*srcH*4, 0.0f);
    Buf tmp    ((size_t)srcW*srcH*4);
    double ps_total = 0.0;

    ps_total += accumulate_stage(
        src, srcW, srcH, lensFi, lensPS,
        lensSz, lensWm, pixelUm, sq_str,
        lensCI, lensCS, lensHI, lensHS, lensBI, lensBS,
        chroma, psf_acc, tmp);

    ps_total += accumulate_stage(
        src, srcW, srcH, prnFi, prnPS,
        prnSz, prnWm, pixelUm, sq_str,
        prnCI, prnCS, prnHI, prnHS, prnBI, prnBS,
        chroma, psf_acc, tmp);

    int ox=db[0]-sb[0], oy=db[1]-sb[1];
    float fmix=(float)mix, fps=(float)ps_total;

    Buf result((size_t)dstW*dstH*4, 0.0f);
    for (int y=0;y<dstH;++y){
        int sy=y+oy; if(sy<0||sy>=srcH) continue;
        for (int x=0;x<dstW;++x){
            int sx=x+ox; if(sx<0||sx>=srcW) continue;
            size_t si=(size_t)(sy*srcW+sx)*4;
            size_t di=(size_t)(y *dstW+x )*4;
            result[di+0]=src[si+0]+fmix*(psf_acc[si+0]-fps*src[si+0]);
            result[di+1]=src[si+1]+fmix*(psf_acc[si+1]-fps*src[si+1]);
            result[di+2]=src[si+2]+fmix*(psf_acc[si+2]-fps*src[si+2]);
            result[di+3]=src[si+3];
        }
    }

    write_image(result,dd,dstW,dstH,drb,dnc);
    release();
    return kOfxStatOK;
}

static OfxStatus main_entry(const char *action, const void *handle,
                             OfxPropertySetHandle inArgs,
                             OfxPropertySetHandle outArgs)
{
    auto h=(OfxImageEffectHandle)handle;
    if(!strcmp(action,kOfxActionLoad))                             return action_load();
    if(!strcmp(action,kOfxActionUnload))                           return kOfxStatOK;
    if(!strcmp(action,kOfxActionDescribe))                         return action_describe(h);
    if(!strcmp(action,kOfxImageEffectActionDescribeInContext))      return action_describe_in_context(h,inArgs);
    if(!strcmp(action,kOfxActionCreateInstance))                   return kOfxStatOK;
    if(!strcmp(action,kOfxActionDestroyInstance))                  return kOfxStatOK;
    if(!strcmp(action,kOfxImageEffectActionIsIdentity))            return action_is_identity(h,inArgs,outArgs);
    if(!strcmp(action,kOfxImageEffectActionGetRegionsOfInterest))  return action_get_roi(h,inArgs,outArgs);
    if(!strcmp(action,kOfxImageEffectActionRender))                return action_render(h,inArgs);
    return kOfxStatReplyDefault;
}

static void set_host(OfxHost *host){ gHost=host; }

static OfxPlugin gPlugin={
    kOfxImageEffectPluginApi,1,
    kPluginId,kMajorVer,kMinorVer,
    set_host,main_entry
};

EXPORT OfxPlugin *OfxGetPlugin(int nth){ return nth==0?&gPlugin:nullptr; }
EXPORT int OfxGetNumberOfPlugins()     { return 1; }
