
# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)
file(GLOB_RECURSE ops_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cc)

include_directories(${ASCEND_CANN_PACKAGE_PATH}/include)

message("###############" ${ops_srcs})
opbuild(OPS_SRC ${ops_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

add_library(cust_op_proto SHARED ${ops_srcs} ${ASCEND_AUTOGEN_PATH}/op_proto.cc)

target_include_directories(cust_op_proto PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/common
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/slog
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/tikcpp/tikcfw/tiling
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/nnopbase
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/json/include
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime/base                  
        ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/op_common
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_op_proto PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_op_proto PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)
add_library(cust_optiling SHARED ${ops_srcs})

target_include_directories(cust_optiling PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/common
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/slog
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/tikcpp/tikcfw/tiling
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/nnopbase
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/json/include
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime/base                  
        ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/op_common
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_optiling PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_optiling PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
add_library(cust_opapi SHARED ${aclnn_src})
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_opapi PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase)

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)

install(TARGETS cust_op_proto
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
        DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
install(TARGETS cust_optiling
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
install(TARGETS cust_opapi
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
install(FILES ${aclnn_inc}
        DESTINATION packages/vendors/${vendor_name}/op_api/include)
