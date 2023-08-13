# 【腾讯云 Cloud Studio 实战训练营】使用Paddle实现简易深度学习项目，并利用 Flask 搭建 Web服务
## 前言
### 1、腾讯云 Cloud Studio 的背景和基本概念
- Cloud Studio：是基于浏览器的集成式开发环境（IDE），为开发者提供了一个永不间断的云端工作站。用户在使用 Cloud Studio 时无需安装，随时随地打开浏览器就能在线编程。
- 应用场景：
    - 快速启动项目
    - 实时调试网页
    - 远程访问云服务器
- 优点：
    - 脱离本地环境
        - 在线编程，减少在本地搭建环境的复杂和兼容性问题
        - 在云端工作，运行不停歇
        - 方便团队协作，在突发事件下可以利用云端开发及时处理
        - 使用插件，对于感兴趣的代码仓库可以一键克隆
    - 丰富的语言模板
    ![这是图片](image\language-T.png "语言模板")
    - 有趣实用的应用模板
    ![这是图片](image\demo-T.png "应用模板")

### 2、本文的目的

## Cloud Studio 的使用
1. 打开[Cloud Studio](https://cloudstudio.net/),点击注册
    - github
    - coding空间
2. 登录完成后
    - 使用语言模板
    - 新建工作空间
    - 新建自定义模板（本人选用，填写相关信息，完成创建，可以在初始化命令下完成基础环境的搭建（但是我这里没有使用
    ![我的图片](image\self-T.png "自定义模板")

## Paddle实现简易性别辨别
### 1、基础环境的搭建（因为之前没有填写初始化命令
> 这里安装paddle的版本要不高于2.4.2，否则会有0维向量的警告
```
I0813 10:07:55.080526   690 eager_method.cc:140] Warning:: 0D Tensor cannot be used as 'Tensor.numpy()[0]' . In order to avoid this problem, 0D Tensor will be changed to 1D numpy currently, but it's not correct and will be removed in release 2.6. For Tensor contain only one element, Please modify  'Tensor.numpy()[0]' to 'float(Tensor)' as soon as possible, otherwise 'Tensor.numpy()[0]' will raise error in release 2.6.
```
```pip
python -m pip install paddlepaddle==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
> 什么是[paddle](https://www.paddlepaddle.org.cn/)呢？ 一个非常好用的深度学习框架  
飞桨(PaddlePaddle)以百度多年的深度学习技术研究和业务应用为基础，是中国首个自主研发、功能完备、 开源开放的产业级深度学习平台，集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体。目前，飞桨累计开发者535万，服务企业20万家，基于飞桨开源深度学习平台产生了67万个模型。飞桨助力开发者快速实现AI想法，快速上线AI业务。帮助越来越多的行业完成AI赋能，实现产业智能化升级。
### 2、数据读取，模型搭建，损失函数，优化器的选择，开始训练
数据文件格式是每行三个数，分别表示一个人的身高、体重、性别，其中性别1表示男，0表示女。
```python
# 数据读取
class MyDataset(Dataset):
    def __init__(self,label_path,transform=None):
        super(MyDataset,self).__init__()
        self.data_list = []
        with open(label_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').rstrip().split()
                self.data_list.append([np.float32(line[0]),np.float32(line[1]),int(line[2])])
        self.transform = transform

    def __getitem__(self,index):
        data = self.data_list[index]
        return paddle.Tensor(np.array([data[0],data[1]])),paddle.Tensor(np.array(data[2]))

    def __len__(self):
        return len(self.data_list)
```
```python
# 模型构建
class SexNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2,2),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)
```
```python
# 优化器与损失函数
optim = paddle.optimizer.Adam(learning_rate=0.01,weight_decay=1e-3,parameters=model.parameters())
loss_fn = paddle.nn.CrossEntropyLoss()
```
训练代码就不在这里展示，见Github：[]()
![我的图片](image\paddle-Show.png)
## 利用flask搭建web服务
> [flask]()是什么呢? 一个轻量级web开发框架
### 1、基础环境的搭建
```pip
pip install flask
```
### 2、搭建过程
1. 入口文件app.py
```

```
## 整体展示
<video width="640" height="360" controls>  
  <source src="image/show-video.mp4" type="video/mp4">  
  Your browser does not support the video tag.  
</video>

## 个人总结和建议