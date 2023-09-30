# PointBiMssc
## Abstruction

> Point cloud semantic segmentation is a key technique in the digital twin construction of water conservancy project, which can realize the identification and change detection of terrain features. However, constructing full-range point cloud data of the water conservancy environment remains one of the critical challenges for achieving comprehensive digital twin construction. Meanwhile, the openness of the water conservancy environment makes its point cloud data highly complex in structure with indistinct boundaries between different categories and huge differences in volume. This poses challenges to the accuracy and robustness of point cloud semantic segmentation algorithms. Therefore, we adopt unmanned aerial vehicle (UAV)-borne lidar to scan water conservancy scenes and construct a large-scale point cloud dataset, Water Conservancy Segment 3D (WCS3D), with approximately 265 million points. On this basis, we propose a point cloud segmentation model named PointBiMssc based on a bidirectional multi-scale attention mechanism for point cloud semantic segmentation in water conservancy environments. Experiments on our constructed dataset WCS3D and the benchmark dataset ScanNet V2 demonstrate that the proposed PointBiMssc model can accurately accomplish point cloud semantic segmentation tasks and generate high-precision segmentation boundaries. It surpasses the latest models including PointMetaBase, PointConvFormer, SPoTr and PointNeXt on the evaluation metrics of mIoU and OA, achieving state-of-the-art performance.

## Installtion

```bash
conda create -n model python=3.8 -y
conda activate model
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-

cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

```

