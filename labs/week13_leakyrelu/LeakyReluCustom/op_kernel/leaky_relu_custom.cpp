#include "kernel_operator.h"
using namespace AscendC;
class KernelLeakyRelu {
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,uint64_t total,uint32_t tile,float slope){
  total_=total;tile_=tile;slope_=slope;
  x_.SetGlobalBuffer((__gm__ float*)x,total);y_.SetGlobalBuffer((__gm__ float*)y,total);
  pipe_.InitBuffer(qx_,2,tile*sizeof(float));pipe_.InitBuffer(qy_,2,tile*sizeof(float));
  pipe_.InitBuffer(tmp_,tile*sizeof(float));
 }
 __aicore__ inline void Process(){
  const uint64_t tiles=(total_+tile_-1)/tile_;
  for(uint64_t t=GetBlockIdx();t<tiles;t+=GetBlockNum()){
   const uint64_t offset=t*tile_;
   const uint32_t count=total_-offset<tile_?total_-offset:tile_;
   auto x=qx_.AllocTensor<float>();
   DataCopyExtParams cp{1,count*uint32_t(sizeof(float)),0,0,0};
   DataCopyPadExtParams<float> pad{false,0,0,0.0f};
   DataCopyPad(x,x_[offset],cp,pad);qx_.EnQue(x);
   x=qx_.DeQue<float>();auto y=qy_.AllocTensor<float>();auto neg=tmp_.Get<float>();
   // max(x,0) + a*min(x,0) works for any finite nonzero a, including a>1.
   Maxs(y,x,0.0f,count);Mins(neg,x,0.0f,count);PipeBarrier<PIPE_V>();
   Muls(neg,neg,slope_,count);PipeBarrier<PIPE_V>();
   Add(y,y,neg,count);qy_.EnQue(y);qx_.FreeTensor(x);
   y=qy_.DeQue<float>();DataCopyPad(y_[offset],y,cp);qy_.FreeTensor(y);
  }
 }
private:
 TPipe pipe_;TQue<QuePosition::VECIN,2> qx_;TQue<QuePosition::VECOUT,2> qy_;
 TBuf<QuePosition::VECCALC> tmp_;GlobalTensor<float> x_,y_;
 uint64_t total_;uint32_t tile_;float slope_;
};
extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling){
 GET_TILING_DATA(data,tiling);KernelLeakyRelu op;op.Init(x,y,data.totalLength,data.tileLength,data.negativeSlope);op.Process();
}
