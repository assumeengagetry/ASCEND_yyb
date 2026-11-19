#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include "acl/acl.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
extern void matmul_custom_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);
#define CHECK(x) do {auto e=(x);if(e){fprintf(stderr,"ACL error %d %s\n",int(e),#x);return 1;}}while(0)
bool read(const char* p,std::vector<uint8_t>&v){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=v.size()) return false;f.seekg(0);return bool(f.read((char*)v.data(),v.size()));}
int main(int argc,char**argv){
 setvbuf(stdout,nullptr,_IONBF,0);
 if(argc!=7){fprintf(stderr,"Usage: main cores baseM baseN baseK repeats output.bin\n");return 2;}
 int dim=atoi(argv[1]), bm=atoi(argv[2]),bn=atoi(argv[3]),bk=atoi(argv[4]),repeats=atoi(argv[5]);
 if(dim<1||dim>20||repeats<1||bm<=0||bn<=0||bk<=0)return 2;
 constexpr int M=4096,N=5120,K=4096;
 auto platform=platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B4");
 if(!platform)return 3;
 printf("PLATFORM cube_cores=%u vector_cores=%u workspace=%u\n",platform->GetCoreNumAic(),platform->GetCoreNumAiv(),platform->GetLibApiWorkSpaceSize());
 matmul_tiling::MultiCoreMatmulTiling tiler(*platform);
 using namespace matmul_tiling;
 tiler.SetDim(2*dim);
 tiler.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT16);
 tiler.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT16);
 tiler.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT);
 tiler.SetShape(M,N,K);tiler.SetOrgShape(M,N,K);
 tiler.SetFixSplit(bm,bn,-1);tiler.SetBias(false);tiler.SetBufferSpace(-1,-1,-1);
 AscendC::tiling::TCubeTiling t;
 if(tiler.GetTiling(t)<0){fprintf(stderr,"TILING_FAIL\n");return 4;}
 // CANN 8.5.2 does not honor a fixed baseK in SetFixSplit.
 // Reduce the L0 K tile while preserving the generated L1 stage extents.
 int autoBaseK=t.baseK;
 if(t.baseK!=bk){
   if(t.baseK<bk || t.baseK%bk){fprintf(stderr,"UNSUPPORTED_BASE_K_OVERRIDE\n");return 5;}
   int ratio=t.baseK/bk;
   t.baseK=bk;t.stepKa*=ratio;t.stepKb*=ratio;
   t.depthA1*=ratio;t.depthB1*=ratio;
 }
 printf("TILING autoBaseK=%d effectiveBaseK=%d stepKa=%d stepKb=%d depthA1=%d depthB1=%d\n",autoBaseK,t.baseK,t.stepKa,t.stepKb,t.depthA1,t.depthB1);
 printf("CONFIG blockDim=%d usedCoreNum=%d M=%d N=%d K=%d base=%d,%d,%d singleCore=%d,%d,%d repeats=%d warmup=5\n",dim,t.usedCoreNum,M,N,K,t.baseM,t.baseN,t.baseK,t.singleCoreM,t.singleCoreN,t.singleCoreK,repeats);
 if(t.baseM!=bm||t.baseN!=bn||t.baseK!=bk){fprintf(stderr,"TILING_CHANGED_REQUEST\n");return 5;}
 std::filesystem::path resultDir=std::filesystem::path(argv[6]).parent_path();
 std::ofstream jt(resultDir / "tiling.json");
 jt << "{\n\"blockDim\":" << dim << ",\n\"automaticBaseK\":" << autoBaseK;
#define JFIELD(f) jt << ",\n\"" #f "\":" << t.f
 JFIELD(usedCoreNum);JFIELD(M);JFIELD(N);JFIELD(Ka);JFIELD(Kb);
 JFIELD(singleCoreM);JFIELD(singleCoreN);JFIELD(singleCoreK);
 JFIELD(baseM);JFIELD(baseN);JFIELD(baseK);JFIELD(depthA1);JFIELD(depthB1);
 JFIELD(stepM);JFIELD(stepN);JFIELD(stepKa);JFIELD(stepKb);JFIELD(isBias);
 JFIELD(transLength);JFIELD(iterateOrder);JFIELD(shareMode);JFIELD(shareL1Size);JFIELD(shareL0CSize);JFIELD(shareUbSize);JFIELD(dbL0A);JFIELD(dbL0B);JFIELD(dbL0C);
#undef JFIELD
 jt << ",\n\"structureBytes\":" << sizeof(t) << ",\n\"raw_int32_words\":[";
 const int32_t *tw=reinterpret_cast<const int32_t*>(&t);
 for(size_t i=0;i<sizeof(t)/4;++i){if(i)jt<<",";jt<<tw[i];} jt << "]\n}\n";jt.close();
 std::ofstream tb(resultDir / "tiling.bin",std::ios::binary);tb.write(reinterpret_cast<const char*>(&t),sizeof(t));tb.close();
 std::vector<uint8_t> ah(size_t(M)*K*2),bh(size_t(K)*N*2),ch(size_t(M)*N*4);
 if(!read("data/input_a.bin",ah)||!read("data/input_b.bin",bh))return 6;
 CHECK(aclInit(nullptr));CHECK(aclrtSetDevice(0));aclrtStream stream;CHECK(aclrtCreateStream(&stream));
 uint8_t *a,*b,*c,*w,*td;
 CHECK(aclrtMalloc((void**)&a,ah.size(),ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc((void**)&b,bh.size(),ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc((void**)&c,ch.size(),ACL_MEM_MALLOC_HUGE_FIRST));
 size_t ws=platform->GetLibApiWorkSpaceSize();CHECK(aclrtMalloc((void**)&w,ws,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc((void**)&td,sizeof(t),ACL_MEM_MALLOC_HUGE_FIRST));
 CHECK(aclrtMemcpy(a,ah.size(),ah.data(),ah.size(),ACL_MEMCPY_HOST_TO_DEVICE));CHECK(aclrtMemcpy(b,bh.size(),bh.data(),bh.size(),ACL_MEMCPY_HOST_TO_DEVICE));CHECK(aclrtMemcpy(td,sizeof(t),&t,sizeof(t),ACL_MEMCPY_HOST_TO_DEVICE));CHECK(aclrtMemset(c,ch.size(),255,ch.size()));
 for(int i=0;i<5+repeats;++i){matmul_custom_do(dim,stream,a,b,c,w,td);CHECK(aclrtSynchronizeStream(stream));}
 CHECK(aclrtMemcpy(ch.data(),ch.size(),c,ch.size(),ACL_MEMCPY_DEVICE_TO_HOST));
 std::ofstream out(argv[6],std::ios::binary);out.write((char*)ch.data(),ch.size());out.close();if(!out)return 7;
 CHECK(aclrtFree(a));CHECK(aclrtFree(b));CHECK(aclrtFree(c));CHECK(aclrtFree(w));CHECK(aclrtFree(td));CHECK(aclrtDestroyStream(stream));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());
 printf("NPU_MATMUL_SUCCESS elements=%d bytes=%zu output=%s\n",M*N,ch.size(),argv[6]);
}
