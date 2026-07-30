
0. 了解当前目录, 然后重新训练U net attack在三个方法上面, epoch 变成 20 个, constant lr 最后 cosine decay. 修改代码, 然后训练 3个, 在目前的做法上. 
1. 当前方法 minusface 是随机通道, partialface 和 fracface 是固定通道. 给fracface和 partialface 全部加上一个 flag, 是是否指定特定channel, 没记错的话 minusface 和 partialface 的channel 是hard coded 的, 所以加上这个 flag 之后变成随机选取channel, fracface 默认就是随机选取的, 所以我原本加了 seed_everything, 那么这个 flag 就 control 我们要不要去 seed.
2. 之前我们所有的结果(distillation attck)都是在固定 channel 上面的, 接下来完成以下 task: 在fracface和partial 的随机 channel方法上, 重新训练我们的 attack 方法, 然后运行重建, 同样 300 张图, 看看可视化和 Face++, amazon效果. 
3. 把所有 unet 在随机通道上面都训练一次, epoch 和上面一样, 同样看看可视化和 Face++, amazon效果
4. 最终报告 (fixed channe, random channel) cross product (ours, unet attack) cross product (fracface, fracface). minusface 单独报告 ours 和 unet 的方法, 但是没有非随机版本. 2 张可视化图, 一张 fix channel一张 random channel, 第一行原图, 第二行 unet 攻击效果, 第三行我们的攻击效果. fix channel 展示 2 种方法, random 的展示 3 种

你可以用所有 gpu, 自己运行, 有问题 push 我, 结果给我报告结果, **严禁对结果进行分析**
 