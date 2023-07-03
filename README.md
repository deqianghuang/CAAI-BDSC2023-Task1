# CAAI-BDSC社交图谱链接预测任务一代码运行说明
**Requirements**    

torch==1.12.1+cu113  
sklearn==1.2.2  
ipdb==0.13.13  
pandas==2.0.1  
numpy==1.24.3
  
    
**Train**    

下面是两个运用cuda训练RotatE模型和HAKE模型的例子，在codes的上一级文件夹运行如下代码，模型会自动保存在models文件夹里
```
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --model RotatE \
 -n 4096 -b 128 -d 1000 \
 -g 12 -a 1.0 -adv \
 -lr 0.0001 --max_steps 1000000 \
 -save models/RotatE_141 --test_batch_size 2 \

CUDA_VISIBLE_DEVICES=1 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --model HAKE \
 -n 4096 -b 128 -d 1000 \
 -g 12 -a 1.0 -adv \
 -lr 0.0001 --max_steps 1000000 \
 -save models/HAKE_141 --test_batch_size 2 \
```
 **Test&Inference**    
   
根据上一步的Train训练好模型以后，修改codes/ensemble_inference.py文件中281行里的model_list中的元素名为models路径下保存的模型名称,只填一个模型即为单模型预测，多个模型可以多模型预测（预测生成的scores.npy占用内存较大注意磁盘空间）
```
model_list = ["models/HAKE_141","models/RotatE_141"]  #  "models/HAKE_221"
```
接着运行下面的代码即可进行测试，会在当前目录下生成名为H%R221_ensemble.log和可供提交的H%R221_ensemble.json文件    
```
 CUDA_VISIBLE_DEVICES=3 python -u codes/ensemble_inference.py --test_batch_size 2 --exp H%R221_ensemble
 ```
