# sample scripts for running various knowledge distillation approaches
# we use resnet32x4 and resnet8x4 as an example

# CIFAR
# KD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill kd --model_s resnet8x4 -c 1 -d 1 -b 0 --trial 0 --gpu_id 0
# FitNet
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill hint --model_s resnet8x4 -c 1 -d 1 -b 100 --trial 0 --gpu_id 0
# AT
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill attention --model_s resnet8x4 -c 1 -d 1 -b 1000 --trial 0 --gpu_id 0
# SP
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill similarity --model_s resnet8x4 -c 1 -d 1 -b 3000 --trial 0 --gpu_id 0
# VID
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill vid --model_s resnet8x4 -c 1 -d 1 -b 1 --trial 0 --gpu_id 0
# CRD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill crd --model_s resnet8x4 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id 0
# SemCKD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill semckd --model_s resnet8x4 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0
# SRRL
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill srrl --model_s resnet8x4 -c 1 -d 1 -b 1 --trial 0 --gpu_id 0
# SimKD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill simkd --model_s resnet8x4 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0
# mfakd
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill mfakd --model_s resnet8x4 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0
