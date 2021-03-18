import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras.models import Model

def Backbone(input_tensor) -> object:
    model = ResNet101(include_top=False,
                      weights='imagenet',
                      input_tensor=input_tensor,
                      pooling=None)
    conv5_block3_out = model.output #(None, 10, 10, 2048)
    conv6_block1_1_conv = Conv2D(256, kernel_size=(1,1),
                             strides=(2,2),
                             padding='same',
                             name='conv6_block1_1_conv')(conv5_block3_out)
    conv6_block1_1_bn = BatchNormalization(name='conv6_block1_1_bn')(conv6_block1_1_conv)
    conv6_block1_1_relu = ReLU(name='conv6_block1_1_relu')(conv6_block1_1_bn)
    conv6_block1_2_conv = Conv2D(256, kernel_size=(3,3),
                                 padding='same',
                                 name='conv6_block1_2_conv')(conv6_block1_1_relu)
    conv6_block1_2_bn = BatchNormalization(name='conv6_block1_2_bn')(conv6_block1_2_conv)
    conv6_block1_2_relu = ReLU(name='conv6_block1_2_relu')(conv6_block1_2_bn)

    conv6_block1_0_conv = Conv2D(1024, kernel_size=(1,1),
                                 strides=(2,2),
                                 padding='same',
                                 name='conv6_block1_0_conv')(conv5_block3_out)

    conv6_block1_3_conv = Conv2D(1024, kernel_size=(1,1),
                                 padding='same',
                                 name='conv6_block1_3_conv')(conv6_block1_2_relu)
    conv6_block1_0_bn = BatchNormalization(name='conv6_block1_0_bn')(conv6_block1_0_conv)
    conv6_block1_3_bn = BatchNormalization(name='conv6_block1_3_bn')(conv6_block1_3_conv)
    conv6_block1_3_add = Add(name='conv6_block1_3_add')([conv6_block1_0_bn,conv6_block1_3_bn])
    conv6_block1_out = ReLU(name='conv6_block1_out')(conv6_block1_3_add)

    conv7_block1_1_conv = Conv2D(256, kernel_size=(1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 name='conv7_block1_1_conv')(conv6_block1_out)
    conv7_block1_1_bn = BatchNormalization(name='conv7_block1_1_bn')(conv7_block1_1_conv)
    conv7_block1_1_relu = ReLU(name='conv7_block1_1_relu')(conv7_block1_1_bn)
    conv7_block1_2_conv = Conv2D(256, kernel_size=(3, 3),
                                 padding='same',
                                 name='conv7_block1_2_conv')(conv7_block1_1_relu)
    conv7_block1_2_bn = BatchNormalization(name='conv7_block1_2_bn')(conv7_block1_2_conv)
    conv7_block1_2_relu = ReLU(name='conv7_block1_2_relu')(conv7_block1_2_bn)

    conv7_block1_0_conv = Conv2D(1024, kernel_size=(1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 name='conv7_block1_0_conv')(conv6_block1_out)

    conv7_block1_3_conv = Conv2D(1024, kernel_size=(1, 1),
                                 padding='same',
                                 name='conv7_block1_3_conv')(conv7_block1_2_relu)
    conv7_block1_0_bn = BatchNormalization(name='conv7_block1_0_bn')(conv7_block1_0_conv)
    conv7_block1_3_bn = BatchNormalization(name='conv7_block1_3_bn')(conv7_block1_3_conv)
    conv7_block1_3_add = Add(name='conv7_block1_3_add')([conv7_block1_0_bn, conv7_block1_3_bn])
    conv7_block1_out = ReLU(name='conv7_block1_out')(conv7_block1_3_add)

    conv8_block1_conv = Conv2D(512, kernel_size=(3, 3),
                                 padding='valid',
                                 name='conv8_block1_conv')(conv7_block1_out)
    conv8_block1_bn = BatchNormalization(name='conv8_block1_bn')(conv8_block1_conv)
    output = ReLU(name='conv8_block1_out')(conv8_block1_bn)
    return Model(inputs=input_tensor, outputs=output)


