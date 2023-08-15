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
> 这里安装paddle的版本要不高于2.4.2，否则会有0维向量相关的警告
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
1. 首先，我们定义了一个名为SexNet的神经网络模型，该模型包含一个线性层`nn.Linear`（在实际应用中，可能需要更复杂的模型结构来处理更复杂的问题）
2. 然后，我们创建了一个`MyDataset`类，用于从指定的文件路径读取数据。数据包括两个特征（身高，体重）和一个标签（性别）。  
3. 接着，我们定义了训练过程，其中包括了训练和验证数据集、模型、优化器、损失函数和训练的轮数。  
4. 在每个训练轮次中，我们使用数据加载器`DataLoader`从训练数据集中获取数据批次，然后通过模型进行前向传播，计算损失和准确率。然后进行反向传播，更新模型的参数。  
5. 使用的损失函数是交叉熵损失`CrossEntropyLos`。优化器是`Adam`优化器，学习率是0.01，权重衰减是1e-3。
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
训练代码就不在这里展示，见Github：[https://github.com/feizai2003/Paddle_sexdemo](https://github.com/feizai2003/Paddle_sexdemo)
![我的图片](image\paddle-Show.png)
## 利用flask搭建web服务
> [flask]()是什么呢? 一个轻量级web开发框架  
> 一般来说一个flask项目会包含三个结构：
> 1. static文件夹：用于存放各种静态文件 css、js、图片等等
> 1. templates文件夹：用于存放html模板文件
> 1. app.py：主文件 ，启动项目需要启动该文件
### 1、基础环境的搭建
```pip
pip install flask
```
### 2、搭建过程
> 其主要功能是接收用户输入的两个数值，然后通过加载之前训练模型（这里由 predict 函数实现）进行处理，最后将结果返回给用户
> 1. 从 flask 导入 Flask 类，以及 render_template 和 request 函数，以及所需的numpy库
> 2. 从 demo.py 中导入 init_weight 和 predict 函数
> 3. 创建一个 Flask 实例，名称为 app
> 4. 使用 init_weight 函数初始化权重，并将结果存储在变量 layer 中
> 5. 定义了两个路由。@app.route('/') 表示当用户访问根路径 ('/') 时，会执行 home 函数并返回渲染的 'index.html' 模板
> 6. @app.route('/calculate', methods=['POST']) 表示当用户访问 '/calculate' 路径且使用 POST 方法时，会执行 calculate 函数。
> 7. calculate 函数首先获取 'input1' 和 'input2' 两个表单中的值（这些值由用户在web界面输入），然后使用 predict 函数对这两个值进行处理，并将结果存储在 output 中。
如果 output 的第一个元素为0，那么将结果设为 "女"，否则设为 "男"。
> 8. 最后，将处理后的输入（之前输入的数字会显示在输入框下的）和结果作为参数传递给 'index.html' 模板进行渲染，然后返回该模板。
1. 入口文件app.py
```python
from flask import Flask, render_template, request
from demo import init_weight,predict
import numpy as np
app = Flask(__name__)
layer = init_weight()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    input1 = request.form.get('input1')
    input2 = request.form.get('input2')
    output = predict(layer,np.float32(input1),np.float32(input2))
    if output[0] == 0:
        output = "女"
    else:
        output = "男"
    return render_template('index.html',input1=input1,input2=input2,result=output)


if __name__ == '__main__':
    app.run(debug=True)
```
2. index.html展示
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Calculator</title>
</head>
<body>
    <form method="POST" action="/calculate">
        <input type="text" name="input1" placeholder="身高（米）">
        <div id="output1">{{ input1 }}</div>
        <input type="text" name="input2" placeholder="体重（千克）">
        <div id="output2">{{ input2 }}</div>
        <button type="submit" name="button1">Calculate</button>
    </form>
    <div>{{ result }}</div>
</body>
</html>
```

## 整体展示
<video width="640" height="360" controls>  
  <source src="image/show-video.mp4" type="video/mp4">  
  Your browser does not support the video tag.  
</video>

## 个人总结和建议
### 总结：
使用了Cloud Studio之后，发现这个一个非常好的平台，随时随地就可以开发，我这次还体验和尝试了使用平板进行代码开发，体验还是挺不错的，之前总是远程自己的电脑进行开发，现在选择不仅可以云端，同时也可以用ubuntu的开发环境，熟悉一下linux也是不错的选择。内置的网页，更是方便了我调试web应用。同时还具有免费的额度，我在慢慢靠近将这作为自己的第二开发形式。此外还要感谢腾讯云举办的这个活动，让我亲身体验了一把。
### 建议：
1. 自定义模板生成后，还可以进行一些配置修改
2. 丰富高亮显示：Python高亮显示对于一些导入之后没有用到的库，是没有提示的