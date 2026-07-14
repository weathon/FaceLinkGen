from functools import partial
from .model_resnet import ResNet_50, ResNet_101, ResNet_152
from .model_irse import IR_18, IR_34, IR_50, IR_101, IR_152, IR_200
from .model_irse import IR_SE_50, IR_SE_101, IR_SE_152, IR_SE_200
from .model_mobilefacenet import MobileFaceNet
from .model_efficientnet import EfficientNetB0, EfficientNetB1
from .model_ghostnet import GhostNet
from .fbnets.fbnet_builder import get_fbnet_model

_model_dict = {
    'ResNet_50': ResNet_50,
    'ResNet_101': ResNet_101,
    'ResNet_152': ResNet_152,
    'IR_18': IR_18,
    'IR_34': IR_34,
    'IR_50': IR_50,
    'IR_101': IR_101,
    'IR_152': IR_152,
    'IR_200': IR_200,
    'IR_SE_50': IR_SE_50,
    'IR_SE_101': IR_SE_101,
    'IR_SE_152': IR_SE_152,
    'IR_SE_200': IR_SE_200,
    'MobileFaceNet': MobileFaceNet,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'GhostNet': GhostNet,
    'fbnet_a': partial(get_fbnet_model, "fbnet_a"),
    'fbnet_b': partial(get_fbnet_model, "fbnet_b"),
    'fbnet_c': partial(get_fbnet_model, "fbnet_c"),
}



def get_model(key, input_size=(112, 112), input_channel=81, **kwargs):
    """ 
    根据指定的 key 返回相应的模型。
    
    Args:
        key (str): 选择模型的关键字，如 'ir_18', 'ir_34', 'ir_50' 等
        input_size (tuple): 输入图像的尺寸，默认为 (112, 112)
        input_channel (int): 输入通道数，默认为 81
        **kwargs: 其他可选参数，如是否使用 `ir_se` 等
    """
    
    def create_model(key):
        if key == 'ir_18':
            return IR_18(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_34':
            return IR_34(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_50':
            return IR_50(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_101':
            return IR_101(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_152':
            return IR_152(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_200':
            return IR_200(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_se_50':
            return IR_SE_50(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_se_101':
            return IR_SE_101(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_se_152':
            return IR_SE_152(input_size, input_channel=input_channel, **kwargs)
        elif key == 'ir_se_200':
            return IR_SE_200(input_size, input_channel=input_channel, **kwargs)
        else:
            raise ValueError(f"Model key '{key}' not recognized. Please choose from ['ir_18', 'ir_34', 'ir_50', 'ir_101', 'ir_152', 'ir_200', 'ir_se_50', 'ir_se_101', 'ir_se_152', 'ir_se_200'].")
    
    return create_model(key)




# def get_model(key):
#     """ Get different backbone network by key,
#         support ResNet50, ResNet_101, ResNet_152
#         IR_18, IR_34, IR_50, IR_101, IR_152, IR_200,
#         IR_SE_50, IR_SE_101, IR_SE_152, IR_SE_200,
#         EfficientNetB0, EfficientNetB1.
#         MobileFaceNet, FBNets.
#     """
#     if key in _model_dict.keys():
#         return _model_dict[key]
#     else:
#         raise KeyError("not support model {}".format(key))


