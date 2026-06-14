// Nimbus Diffusion
// Copyright (C) 2026 Mohamed Mabrok
// GPL v3 - see LICENSE
//
// OFX diffusion plugin for DaVinci Resolve. Lens and Print stages,
// anamorphic stretch, chromatic bloom.
//
// PSF math based on spektrafilm by Andrea Volpato (GPLv3).
// C++ port: Gaussian-mixture surrogate of the 2D isotropic exponential PSF
// via Young-van Vliet IIR — matches spektrafilm's fast_exponential_filter exactly.

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

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
static const int kMajorVer = 2, kMinorVer = 9;

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
// detail recovery + vignette
#define kParamDetail        "Detail"
#define kParamDetailRadius  "DetailRadius"
#define kParamVignette      "Vignette"
#define kParamVigFeather    "VignetteFeather"

// filter presets, from spektrafilm's characterization
static const int MAX_SUB = 4;

struct GroupCfg { double lambda_um, spread; int n; double alpha; };

struct FamilyCfg {
    const char *id, *label;
    GroupCfg core, halo, bloom;
    double w_c, w_h, w_b, warmth_base, gain;
};

static const FamilyCfg kFamilies[8] = {
    { "glimmerglass",        "Glimmerglass",
      {  10.0,1.5,2,0.0},{  50.0,2.0,3,0.0},{  260.0,2.5,4,3.2},
      0.60,0.30,0.10, 0.00,0.65 },
    { "black_pro_mist",      "Black Pro-Mist",
      {  16.0,1.5,2,0.0},{  95.0,2.0,3,0.0},{  380.0,2.5,4,3.5},
      0.40,0.47,0.13, 0.65,0.75 },
    { "pro_mist",            "Pro-Mist",
      {  14.0,1.5,2,0.0},{ 150.0,2.0,3,0.0},{  650.0,2.5,4,2.9},
      0.28,0.42,0.30, 0.40,1.05 },
    { "cinebloom",           "CineBloom",
      {  20.0,1.5,2,0.0},{ 200.0,2.0,3,0.0},{ 1000.0,2.5,4,2.5},
      0.22,0.30,0.48, 0.85,1.00 },
    // --- new filters ---
    // Hollywood Black Magic: tight precise highlight glow, almost no bloom.
    { "hollywood_black_magic","Hollywood Black Magic",
      {   8.0,1.5,2,0.0},{  45.0,2.0,3,0.0},{  180.0,2.5,4,2.8},
      0.70,0.25,0.05, 0.15,0.50 },
    // Supermist: heavy all-over diffusion, strong bloom.
    { "supermist",           "Supermist",
      {  18.0,1.5,2,0.0},{ 110.0,2.0,3,0.0},{  750.0,2.5,4,2.7},
      0.20,0.35,0.45, 0.55,1.20 },
    // White Pro-Mist: same spread as Pro-Mist, but cooler and cleaner.
    { "white_pro_mist",      "White Pro-Mist",
      {  13.0,1.5,2,0.0},{ 130.0,2.0,3,0.0},{  580.0,2.5,4,2.9},
      0.32,0.44,0.24,-0.10,0.95 },
    // Black Diffusion/FX: widest, softest spread — more than CineBloom.
    { "black_diffusion_fx",  "Black Diffusion/FX",
      {  25.0,1.5,2,0.0},{ 280.0,2.0,3,0.0},{ 1400.0,2.5,4,2.4},
      0.15,0.25,0.60, 0.70,1.10 },
};
static const int kNumFamilies = 8;

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

// Isotropic Gaussian blur via 3-pass box filter (separable H then V).
// 3 box-blur passes converge to Gaussian by the CLT; sigma ≈ r per pass.
// Separable Gaussian in 2D is perfectly isotropic (circular, not diamond).
// sigma_h/sigma_v: per-channel Gaussian sigma in pixels (can differ for anamorphic).
// Alpha channel is left untouched.
using Buf = std::vector<float>;

static void gauss_blur_from(const Buf &src, Buf &dst, int w, int h,
                              const double sigma_h[3], const double sigma_v[3])
{
    // Per-channel box radius: 3 passes of width (2r+1) give sigma ≈ sqrt(r*(r+1)).
    // For large r: sigma ≈ r. For r=0: no blur.
    int rh[3], rv[3];
    for (int c=0;c<3;++c){
        rh[c] = (sigma_h[c]>=0.5) ? std::max(1,(int)std::round(sigma_h[c])) : 0;
        rv[c] = (sigma_v[c]>=0.5) ? std::max(1,(int)std::round(sigma_v[c])) : 0;
    }

    dst.resize((size_t)w*h*4);
    std::copy(src.begin(), src.end(), dst.begin());

    // --- 3 horizontal box-blur passes (parallelized over rows) ---
    #pragma omp parallel
    {
        std::vector<float> buf(w), tmp(w);
        #pragma omp for schedule(static)
        for (int y=0; y<h; ++y){
            float *row = dst.data()+(size_t)y*w*4;
            for (int c=0; c<3; ++c){
                int r = rh[c]; if (r==0) continue;
                double inv = 1.0/(2*r+1);
                for (int x=0;x<w;++x) buf[x]=row[x*4+c];
                for (int pass=0; pass<3; ++pass){
                    // clamped-edge running sum
                    double sum=0;
                    for (int k=-r;k<=r;++k)
                        sum+=buf[std::max(0,std::min(w-1,k))];
                    for (int x=0;x<w;++x){
                        tmp[x]=(float)(sum*inv);
                        sum+=buf[std::min(w-1,x+r+1)]-buf[std::max(0,x-r)];
                    }
                    std::swap(buf,tmp);
                }
                for (int x=0;x<w;++x) row[x*4+c]=buf[x];
            }
        }
    }

    // --- 3 vertical box-blur passes (parallelized over columns) ---
    #pragma omp parallel
    {
        std::vector<float> col(h), tmp(h);
        #pragma omp for schedule(static)
        for (int x=0; x<w; ++x){
            for (int c=0; c<3; ++c){
                int r = rv[c]; if (r==0) continue;
                double inv = 1.0/(2*r+1);
                for (int y=0;y<h;++y) col[y]=dst[(size_t)(y*w+x)*4+c];
                for (int pass=0; pass<3; ++pass){
                    double sum=0;
                    for (int k=-r;k<=r;++k)
                        sum+=col[std::max(0,std::min(h-1,k))];
                    for (int y=0;y<h;++y){
                        tmp[y]=(float)(sum*inv);
                        sum+=col[std::min(h-1,y+r+1)]-col[std::max(0,y-r)];
                    }
                    std::swap(col,tmp);
                }
                for (int y=0;y<h;++y) dst[(size_t)(y*w+x)*4+c]=col[y];
            }
        }
    }
}

// ─── Young-van Vliet 3rd-order IIR Gaussian (spektrafilm's fast_gaussian_filter) ─
//
// 3-term Gaussian mixture that approximates exp(-r/λ)/(2πλ²):
//   exp(-r/λ) ≈ 0.1633·G(0.5360λ) + 0.6496·G(1.5236λ) + 0.1870·G(2.7684λ)
// Coefficients taken verbatim from spektrafilm's fast_exponential_filter.
static const double kExpFitA[3] = {0.1633, 0.6496, 0.1870};
static const double kExpFitS[3] = {0.5360, 1.5236, 2.7684};

// Compute YvV 3rd-order IIR coefficients for Gaussian with given sigma.
static void yvv_coeffs(double sigma, double &B, double &b1, double &b2, double &b3){
    double q = (sigma >= 2.5) ? 0.98711*sigma - 0.96330
                              : 3.97156 - 4.14554*std::sqrt(std::max(0.0, 1.0-0.26891*sigma));
    double q2=q*q, q3=q2*q;
    double b0  = 1.57825 + 2.44413*q + 1.4281*q2 + 0.422205*q3;
    double bc1 = 2.44413*q + 2.85619*q2 + 1.26661*q3;
    double bc2 = -(1.4281*q2 + 1.26661*q3);
    double bc3 = 0.422205*q3;
    B  = 1.0-(bc1+bc2+bc3)/b0; b1=bc1/b0; b2=bc2/b0; b3=bc3/b0;
}

// Forward+backward YvV IIR Gaussian along rows (H), in-place on channel ch.
static void yvv_h(Buf &buf, int W, int H, int ch,
                  double B, double b1, double b2, double b3){
    #pragma omp parallel
    {
        std::vector<double> t(W);
        #pragma omp for schedule(static)
        for(int y=0;y<H;++y){
            float *row=buf.data()+y*W*4;
            double w1=row[ch],w2=row[ch],w3=row[ch];
            for(int x=0;x<W;++x){
                double w=B*row[x*4+ch]+b1*w1+b2*w2+b3*w3;
                t[x]=w; w3=w2; w2=w1; w1=w;
            }
            double y1=t[W-1],y2=t[W-1],y3=t[W-1];
            for(int x=W-1;x>=0;--x){
                double yv=B*t[x]+b1*y1+b2*y2+b3*y3;
                t[x]=yv; y3=y2; y2=y1; y1=yv;
            }
            for(int x=0;x<W;++x) row[x*4+ch]=(float)t[x];
        }
    }
}

// Forward+backward YvV IIR Gaussian along columns (V), in-place on channel ch.
static void yvv_v(Buf &buf, int W, int H, int ch,
                  double B, double b1, double b2, double b3){
    #pragma omp parallel
    {
        std::vector<double> col(H), out(H);
        #pragma omp for schedule(static)
        for(int x=0;x<W;++x){
            for(int y=0;y<H;++y) col[y]=buf[(y*W+x)*4+ch];
            double w1=col[0],w2=col[0],w3=col[0];
            for(int y=0;y<H;++y){
                double w=B*col[y]+b1*w1+b2*w2+b3*w3;
                out[y]=w; w3=w2; w2=w1; w1=w;
            }
            double y1=out[H-1],y2=out[H-1],y3=out[H-1];
            for(int y=H-1;y>=0;--y){
                double yv=B*out[y]+b1*y1+b2*y2+b3*y3;
                out[y]=yv; y3=y2; y2=y1; y1=yv;
            }
            for(int y=0;y<H;++y) buf[(y*W+x)*4+ch]=(float)out[y];
        }
    }
}

// YvV IIR Gaussian blur on channel ch with (optionally anisotropic) sigma_h / sigma_v.
static void yvv_blur_ch(Buf &buf, int W, int H, int ch,
                         double sigma_h, double sigma_v){
    if(sigma_h>=0.5){
        double B,b1,b2,b3; yvv_coeffs(sigma_h,B,b1,b2,b3);
        yvv_h(buf,W,H,ch,B,b1,b2,b3);
    }
    if(sigma_v>=0.5){
        double B,b1,b2,b3; yvv_coeffs(sigma_v,B,b1,b2,b3);
        yvv_v(buf,W,H,ch,B,b1,b2,b3);
    }
}
// ─────────────────────────────────────────────────────────────────────────────

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

// Accumulate one diffusion stage into psf_acc using the Gaussian-mixture surrogate
// of the 2D isotropic exponential PSF — identical to spektrafilm's fast_exponential_filter:
//   exp(-r/λ) ≈ Σ_g kExpFitA[g] · YvV_Gaussian(kExpFitS[g]·λ)
// Applied separably (H then V) via Young-van Vliet 3rd-order IIR — O(1) per pixel.
// Result: psf_acc += p_s * (K * src)
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
    if(p_s<=0.0) return 0.0;
    const FamilyCfg &fam=kFamilies[fi];
    double wc=std::max(0.0,fam.w_c*coreI), wh=std::max(0.0,fam.w_h*haloI), wb=std::max(0.0,fam.w_b*bloomI);
    double wt=wc+wh+wb; if(wt<=0.0) return 0.0;
    wc/=wt; wh/=wt; wb/=wt;

    double eff_w=fam.warmth_base+warmth;
    GroupCfg cg={fam.core.lambda_um*coreSz, fam.core.spread, fam.core.n, fam.core.alpha};
    GroupCfg hg={fam.halo.lambda_um*haloSz, fam.halo.spread, fam.halo.n, fam.halo.alpha};
    GroupCfg bg={fam.bloom.lambda_um*bloomSz,fam.bloom.spread,fam.bloom.n,fam.bloom.alpha};
    double cl[MAX_SUB],cw_[MAX_SUB],hl[MAX_SUB],hw[MAX_SUB],bl[MAX_SUB],bw_[MAX_SUB];
    expand_group(cg,false,cl,cw_); expand_group(hg,false,hl,hw); expand_group(bg,true,bl,bw_);
    double hch[3][MAX_SUB]; halo_channel_weights(hw,hg.n,eff_w,hch);
    const double cs[3]={1.0+chroma, 1.0, std::max(0.1, 1.0-chroma*0.5)};

    float ps_f=(float)p_s;
    size_t np=(size_t)W*H;
    tmp.resize(np*4);

    // ── Core: achromatic — same lambda for all channels ──────────────────────
    for(int k=0;k<cg.n;++k){
        if(cw_[k]<=0.0) continue;
        double lam=cl[k]*size/pixelUm;
        for(int g=0;g<3;++g){
            double sh=kExpFitS[g]*lam*sq_str, sv=kExpFitS[g]*lam/sq_str;
            std::copy(src.begin(),src.end(),tmp.begin());
            for(int ch=0;ch<3;++ch) yvv_blur_ch(tmp,W,H,ch,sh,sv);
            float coeff=ps_f*(float)(wc*cw_[k]*kExpFitA[g]);
            for(size_t i=0;i<np;++i){
                psf_acc[i*4+0]+=coeff*tmp[i*4+0];
                psf_acc[i*4+1]+=coeff*tmp[i*4+1];
                psf_acc[i*4+2]+=coeff*tmp[i*4+2];
            }
        }
    }

    // ── Halo: per-channel warmth weights, same spatial scale ─────────────────
    for(int k=0;k<hg.n;++k){
        double lam=hl[k]*size/pixelUm;
        for(int g=0;g<3;++g){
            double sh=kExpFitS[g]*lam*sq_str, sv=kExpFitS[g]*lam/sq_str;
            std::copy(src.begin(),src.end(),tmp.begin());
            for(int ch=0;ch<3;++ch) yvv_blur_ch(tmp,W,H,ch,sh,sv);
            for(int ch=0;ch<3;++ch){
                float coeff=ps_f*(float)(wh*hch[ch][k]*kExpFitA[g]);
                for(size_t i=0;i<np;++i) psf_acc[i*4+ch]+=coeff*tmp[i*4+ch];
            }
        }
    }

    // ── Bloom: per-channel chromatic scale — different lambda per channel ─────
    for(int k=0;k<bg.n;++k){
        if(bw_[k]<=0.0) continue;
        double lam=bl[k]*size/pixelUm;
        for(int g=0;g<3;++g){
            std::copy(src.begin(),src.end(),tmp.begin());
            for(int ch=0;ch<3;++ch){
                double sh=kExpFitS[g]*lam*cs[ch]*sq_str, sv=kExpFitS[g]*lam*cs[ch]/sq_str;
                yvv_blur_ch(tmp,W,H,ch,sh,sv);
            }
            float coeff=ps_f*(float)(wb*bw_[k]*kExpFitA[g]);
            for(size_t i=0;i<np;++i){
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
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,0,"Glimmerglass");          \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,1,"Black Pro-Mist");       \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,2,"Pro-Mist");             \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,3,"CineBloom");            \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,4,"Hollywood Black Magic");\
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,5,"Supermist");            \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,6,"White Pro-Mist");       \
    gPropSuite->propSetString(pProps,kOfxParamPropChoiceOption,7,"Black Diffusion/FX");   \
    gPropSuite->propSetString(pProps,kOfxParamPropParent,      0,GRP_);                   \
    DEF_D(STR_,"Strength",                                                        \
        "How strong the filter is. Matches commercial stop ratings: 0.125=1/8, 0.25=1/4, 0.5=1/2, 1.0=full, 2.0=heavy.", \
        STR_DEF_, 0.0,2.0, 0.0,2.0, GRP_)                                        \
    DEF_D(SZ_,"Size","Scale all blur radii up or down.",                          \
        1.0, 0.1,4.0, 0.1,4.0, GRP_)                                             \
    DEF_D(WM_,"Warmth","Tint the halo. +1 = warm yellowish glow, -1 = cool blue.", \
        0.0, -1.5,1.5, -1.5,1.5, GRP_)

// 6 fine-tuning controls per stage — PFX_ is "Lens" or "Print"
#define DEF_STAGE_ADV(CI_,CS_,HI_,HS_,BI_,BS_, CI_D,CS_D,HI_D,HS_D,BI_D,BS_D, PFX_, GRP_) \
    DEF_D(CI_, PFX_ " Core Intensity","Core brightness multiplier.",                         \
        CI_D, 0.0,4.0, 0.0,4.0, GRP_)                                                       \
    DEF_D(CS_, PFX_ " Core Size","Core radius multiplier.",                                  \
        CS_D, 0.1,4.0, 0.1,4.0, GRP_)                                                       \
    DEF_D(HI_, PFX_ " Halo Intensity","Halo brightness multiplier.",                         \
        HI_D, 0.0,4.0, 0.0,4.0, GRP_)                                                       \
    DEF_D(HS_, PFX_ " Halo Size","Halo radius multiplier.",                                  \
        HS_D, 0.1,4.0, 0.1,4.0, GRP_)                                                       \
    DEF_D(BI_, PFX_ " Bloom Intensity","Bloom brightness multiplier.",                       \
        BI_D, 0.0,4.0, 0.0,4.0, GRP_)                                                       \
    DEF_D(BS_, PFX_ " Bloom Size","Bloom radius multiplier.",                                \
        BS_D, 0.1,4.0, 0.1,4.0, GRP_)

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

    DEF_D(kParamDetail, "Detail Recovery",
        "Recovers sharpness lost to diffusion — adds back an unsharp mask of the source. "
        "0=off, 1=maximum. Use Detail Radius to pick the texture frequency.",
        0.0, 0.0,1.0, 0.0,1.0, "")
    DEF_D(kParamDetailRadius, "Detail Radius",
        "Blur radius (pixels) used for detail recovery. "
        "Smaller = sharpen fine grain/texture; larger = recover broader structure.",
        1.5, 0.5,4.0, 0.5,4.0, "")

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

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"VignetteGroup",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,         0,"Vignette");
    gPropSuite->propSetInt   (pProps,kOfxParamPropGroupOpen,0,0);

    DEF_D(kParamVignette, "Amount",
        "How much to darken the edges. 0=off, 1=full black corners.",
        0.0, 0.0,1.0, 0.0,1.0, "VignetteGroup")
    DEF_D(kParamVigFeather, "Feather",
        "How gradually the vignette falls off toward the center. "
        "Low = hard edge starting near corners; high = very soft, starts from center.",
        0.5, 0.0,1.0, 0.0,1.0, "VignetteGroup")

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"VignetteGroupEnd",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,       0,"");
    gPropSuite->propSetInt   (pProps,kOfxParamPropSecret, 0,1);

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"AdvGroup",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,         0,"Advanced");
    gPropSuite->propSetInt   (pProps,kOfxParamPropGroupOpen,0,0);

    DEF_D(kParamPixelSize,"Sensor Pixel um",
        "Physical pixel size in micrometers. 6.0 is typical for a cinema camera. "
        "Affects the absolute scale of the blur.",
        12.4, 1.0,24.0, 1.0,24.0, "AdvGroup")

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
        1.0, 1.0, 1.426, 1.0, 1.0, 1.0,
        "Lens", "AdvGroup")

    DEF_STAGE_ADV(
        kParamPrintCore,  kParamPrintCoreSize,
        kParamPrintHalo,  kParamPrintHaloSize,
        kParamPrintBloom, kParamPrintBloomSize,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        "Print", "AdvGroup")

    gParamSuite->paramDefine(paramSet,kOfxParamTypeGroup,"AdvGroupEnd",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,       0,"");
    gPropSuite->propSetInt   (pProps,kOfxParamPropSecret, 0,1);

    // about
    gParamSuite->paramDefine(paramSet,kOfxParamTypeString,"AboutText",&pProps);
    gPropSuite->propSetString(pProps,kOfxPropLabel,           0,"Info");
    gPropSuite->propSetString(pProps,kOfxParamPropDefault,    0,
        "Nimbus Diffusion v2.9  |  Copyright 2026 Mohamed Mabrok  |  GPL v3 — free & open source");
    gPropSuite->propSetString(pProps,kOfxParamPropStringMode, 0,kOfxParamStringIsLabel);
    gPropSuite->propSetInt   (pProps,kOfxParamPropEnabled,    0,0);

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

    if (gd(kParamVignette,0.0) > 0) return kOfxStatReplyDefault;
    if (gd(kParamDetail,0.0)   > 0) return kOfxStatReplyDefault;
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

    double pixelUm = std::max(1.0, fd(kParamPixelSize,12.4));
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

    double mix      = std::max(0.0,std::min(1.0, gd(kParamMix,      1.0)));
    double pixelUm  = std::max(1e-6,            gd(kParamPixelSize,12.4));
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

    double detail     = std::max(0.0,std::min(1.0, gd(kParamDetail,      0.0)));
    double detailRad  = std::max(0.5,              gd(kParamDetailRadius, 1.5));
    double vignette   = std::max(0.0,std::min(1.0, gd(kParamVignette,    0.0)));
    double vigFeather = std::max(0.01,std::min(1.0,gd(kParamVigFeather,  0.5)));

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
    int ox=db[0]-sb[0], oy=db[1]-sb[1];

    // pass source through if there's nothing to do — avoids a white frame at Strength=0
    bool hasEffect = ((mix>0) && (lensPS>0 || prnPS>0))
                     || (vignette>0.0) || (detail>0.0);
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

    // detail recovery — unsharp mask on source to bring back texture lost to diffusion
    // adds  detail * (src - blur(src, radius))  to the diffused result
    if (detail > 0.0) {
        double dsig = std::max(0.5, detailRad);
        double dlh[3]={dsig,dsig,dsig}, dlv[3]={dsig,dsig,dsig};
        Buf sharp;
        gauss_blur_from(src, sharp, srcW, srcH, dlh, dlv);
        float fd=(float)detail;
        for (int y=0;y<dstH;++y){
            int sy=y+oy; if(sy<0||sy>=srcH) continue;
            for (int x=0;x<dstW;++x){
                int sx=x+ox; if(sx<0||sx>=srcW) continue;
                size_t si=(size_t)(sy*srcW+sx)*4;
                size_t di=(size_t)(y *dstW+x )*4;
                result[di+0]+=fd*(src[si+0]-sharp[si+0]);
                result[di+1]+=fd*(src[si+1]-sharp[si+1]);
                result[di+2]+=fd*(src[si+2]-sharp[si+2]);
            }
        }
    }

    // vignette — smooth radial darkening toward the corners
    if (vignette > 0.0) {
        float cx=(float)srcW*0.5f, cy=(float)srcH*0.5f;
        float fvig=(float)vignette;
        float ffeat=std::max(0.01f,(float)vigFeather);
        for (int y=0;y<dstH;++y){
            int sy=y+oy; if(sy<0||sy>=srcH) continue;
            for (int x=0;x<dstW;++x){
                int sx=x+ox; if(sx<0||sx>=srcW) continue;
                // normalize: (0,0)=center, (±1,±1)=edges, corners at r≈1.41
                float nx=(sx-cx)/cx, ny=(sy-cy)/cy;
                float r=std::sqrt(nx*nx+ny*ny)/1.4142f; // 0=center, 1=corner
                float start=1.0f-ffeat;
                float t=std::max(0.0f,std::min(1.0f,(r-start)/ffeat));
                t=t*t*(3.0f-2.0f*t); // smoothstep — no harsh edge
                float v=1.0f-fvig*t;
                size_t di=(size_t)(y*dstW+x)*4;
                result[di+0]*=v;
                result[di+1]*=v;
                result[di+2]*=v;
            }
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
