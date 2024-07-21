### DPO(Direct Preference Optimization)

#### 原理 
1. 


#### 数据
1. 格式:
- 举例: {"question":'...',"accept":"...","reject":"..."}
```python


```


### 模型显存占用 

1. 计算公式 
> [!NOTE]
> float32类型占用4个字节。所以一个粗略的计算方法就是，每10亿个参数，占用4G显存(实际应该是10^9*4/1024/1024/1024=3.725G，为了方便可以记为4G)。半精度的FP16/BF16来加载，这样每个参数只占2个字节，所需显存就降为一半.

```bash
#LLaMA的参数量为7000559616，那么全精度加载这个模型参数需要的显存为：
7000559616 * 4 /1024/1024/1024 = 26.08G
```
  - 具体参数表

  | dtype | 每10 亿参数需要占用内存 |
  | --------------- | --------------- | 
  | float32 | 4G |
  | fp16/bf16 | 2G | 
  | int8 | 1G | 
  | int4 | 0.5G |
  

2.具体模型显存占用量
| Model | GPU Usage |
| --------------- | --------------- | 
| Qwen-7B-fp16 | 13.04G | 
| Qwen-7B-Int8 | 6.52G | 

### 训练显存使用之处 
1. 加载模型
2. 优化器(optimizer states)
3. 激活函数(activations)
```bash
https://blog.csdn.net/guojiajiajiu/article/details/138466134
https://www.bilibili.com/read/cv35386797/
```

### 训练经验和注意事项 
> [!NOTE]
> 如果用带有history的数据训练base模型，需要指定支持多轮对话的template(base模型往往不支持多轮对话)，对于这种情况我们默认设置了chatmltemplate，你也可以支持--model_type 来选择训练模型的template
我们默认在训练时设置--gradient_checkpointing true来节约显存, 这会略微降低训练速度.
如果你使用的是V100等较老的GPU, 你需要设置--dtype AUTO或者--dtype fp16, 因为其不支持bf16.
如果你的机器是A100等高性能显卡, 且使用的是qwen系列模型, 推荐你安装flash-attn, 这将会加快训练和推理的速度以及显存占用(3090, V100等显卡不支持flash-attn进行训练). 支持flash-attn的模型可以查看LLM支持的模型

1. 训练过程中显存OOM
```bash
模型：LLaMA2-7B-SFT-FULL
设备：单机8卡A100-80G

FULL
全量微调的LLaMA2
使用Deepspeed ZeRO3配置会出现weight-2D xxxxx 报错，issue里面用
后面改用Deepspeed ZeRO2 8卡会出现显存OOM

后面修改模式为LoRA
使用ZeRO2配置，可以进行训练，但是在训练过程中会出现显存OOM报错
监控显卡平均显存占用在60G左右，部分单卡到75G左右

有两个疑问：
DPO要双倍显存指的是什么？SFT全量大约需要16-20倍参数量的显存，那DPO就要32-40倍吗？
当前的DPO是不是和deepspeed的配置不太兼容？要用accelerate来启动项目是吗？
```

2. DPO训练会加载两个模型 
```bash
由于人类对齐训练在一张卡上加载两个模型，因此比微调的显存多占用一个推理模型的显存使用量。
```
