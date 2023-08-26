本项目来源于[《腾讯云 Cloud Studio 实战训练营》](https://marketing.csdn.net/p/06a21ca7f4a1843512fa8f8c40a16635)的参赛作品，该作品在腾讯云 [Cloud Studio](https://www.cloudstudio.net/?utm=csdn) 中运行无误。

# 项目说明
这是一个用 IDE [Cloud Studio](https://www.cloudstudio.net/?utm=csdn) 快速搭建，并开发一个简易的深度学习项目（通过身高和体重分辨男女，当然只是一个演示项目），通过paddlepaddle进行模型开发与训练，然后利用flask搭建一个web平台，具有更好的交互性。从 0 到 1 体验云 IDE 给我们带来的优势，不需要装各种环境，简单易用，开箱即可上手。
## 相关技术栈
Python + HTML + Flask + PaddlePaddle
## 项目运行
```
python -m pip install paddlepaddle==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install flask
# 训练模型并保存
python demo.py

# 部署到web
python app.py
```

## TODO
- [ ] 增加gradio
- [ ] 增加Pytorch搭建和训练模型
- [ ] 增加Mindspore搭建和训练模型
- [ ] 增加多种模型部署
