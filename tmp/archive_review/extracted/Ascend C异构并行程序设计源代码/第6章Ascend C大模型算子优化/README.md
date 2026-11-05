# 安装开发套件包  
1. 获取CANN软件包和communitysdk包（与华为生态工程师获取）。
当前配套的版本链接为：  
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN_Test/20231215/Ascend-cann-toolkit_CANN-7.1_linux-x86_64.run  
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN_Test/20231215/Ascend-cann-toolkit_CANN-7.1_linux-aarch64.run  
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN_Test/20231215/Ascend-cann-communitysdk_CANN-7.1_linux-x86_64.run  
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN_Test/20231215/Ascend-cann-communitysdk_CANN-7.1_linux-aarch64.run  


2. 安装CANN软件包，安装完成后，CANN开发套件的相关组件默认存储在“/usr/local/Ascend/ascend-toolkit/latest”路径下。  
 x86_64  
./Ascend-cann-toolkit_{software version}_linux-x86_64.run --install   
 aarch64  
./Ascend-cann-toolkit_{software version}_linux-aarch64.run --install

3. 安装communitysdk包（临时）,路径与上线的安装目录一致，后缀加opensdk  
  x86_64  
 ./Ascend-cann-communitysdk_{software version}_linux-x86_64.run --noexec --extract=/usr/local/Ascend/ascend-toolkit/latest/opensdk  
  aarch64   
 ./Ascend-cann-communitysdk_{software version}_linux-aarch64.run --noexec --extract=/usr/local/Ascend/ascend-toolkit/latest/opensdk
 
4. （可选）使用自定义路径安装，其中{install_path}为指定的安装路径，安装完成后，CANN开发套件的相关组件存储在“${install_path}/ascend-toolkit/latest”路径下。  
 x86_64  
./Ascend-cann-toolkit_{software version}_linux-x86_64.run --install --install-path={install_path}  
 aarch64  
./Ascend-cann-toolkit_{software version}_linux-aarch64.run --install --install-path={install_path}  

# 算子编译与安装  
基于AscendC，提供了一个从用户角度开发的自定义算子，下面将介绍如何编译和安装Flash Attention算子包，并在Ascend910B芯片设备上运行。  
1. 下载算子源码工程，下载地址：https://gitee.com/cann/ascend-attention  
2. 设置CANN运行环境变量（路径参考第一步**安装开发套件包**中安装的路径）  
  source /usr/local/Ascend/ascend-toolkit/set_env.sh  
  如果指定了安装路径，则执行source <install_path>/ascend-toolkit/set_env.sh  

3. 配置编译依赖CANN包路径  
  进入ops_ascendc目录打开CMakePreset.json，编辑`ASCEND_CANN_PACKAGE_PATH`为软件包安装路径的latest目录，例如/usr/local/Ascend/ascend-toolkit/latest  
  
4. 指定安装目录（软件包安装路径的latest目录），编译自定义算子包  
  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  进入ops_ascendc目录执行./build.sh  
5. 安装自定义算子包  
  （x86_64）  
  ./build_out/custom_opp_ubuntu_x86_64.run --quiet  
  （aarch64）  
  ./build_out/custom_opp_euleros_aarch64.run --quiet  


# 失败FAQ

失败现象：

'''
More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
Traceback (most recent call last):
  File "/usr/local/Ascend/ascend-toolkit/CANN-7.1/python/site-packages/opc_tool/opc.py", line 24, in <module>
    import ge_ir_pb2
  File "/usr/local/Ascend/ascend-toolkit/CANN-7.1/python/site-packages/opc_tool/ge_ir_pb2.py", line 33, in <module>
    _descriptor.EnumValueDescriptor(
/home/CI_DATA/attention/build_out/op_kernel/binary/ascend910b/bin/flash_attention_score/FlashAttentionScore_3a041239c1f2c5cf9f490f48d2bb47db.json not generated!
  File "/usr/local/lib/python3.8/dist-packages/google/protobuf/descriptor.py", line 789, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
'''

解决方式：
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
