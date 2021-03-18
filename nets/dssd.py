import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
#from keras.layers import AtrousConvolution2D
from tensorflow.keras.layers import Conv2D, \
    Convolution2DTranspose, ReLU, Multiply, BatchNormalization, Add, Flatten, Input, Reshape, Concatenate
from tensorflow.keras.models import Model
from nets.ResNet101 import Backbone
from nets.ssd_layers import PriorBox
def DSSD320(input_shape, num_classes=21):
    # 320,320,3
    input_tensor = Input(shape=input_shape, name='image_input')
    img_size = (input_shape[1], input_shape[0])

    # 結構
    model = Backbone(input_tensor)
    # -----------------------提取各層網路---------------------------#
    conv3_block4_out = model.get_layer('conv3_block4_out').output #[None, 40, 40, 512]
    conv4_block23_out = model.get_layer('conv4_block23_out').output  # [None, 20, 20, 1024]
    conv5_block3_out = model.get_layer('conv5_block3_out').output  # [None, 10, 10, 2048]
    conv6_block1_out = model.get_layer('conv6_block1_out').output  # [None, 5, 5, 1024]
    conv7_block1_out = model.get_layer('conv7_block1_out').output  # [None, 3, 3, 1024]
    conv8_block1_out = model.get_layer('conv8_block1_out').output  # [None, 1, 1, 512]
    #------------------------進行Deconvolution--------------------------#
    ## dconv9 # [None, 3, 3, 256]
    # Deconvolution layer
    dconv9_dconv = Convolution2DTranspose(filters=256,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          output_padding=(0, 0),
                                          name='dconv9_dconv')(conv8_block1_out)
    dconv9_conv = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         name='dconv9_conv')(dconv9_dconv)
    dconv9_bn = BatchNormalization(name='dconv9_bn')(dconv9_conv)
    # Feature Layer from SSD
    conv9_conv1 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         name='conv9_conv1')(conv7_block1_out)
    conv9_bn1 = BatchNormalization(name='conv9_bn1')(conv9_conv1)
    conv9_relu1 = ReLU(name='conv9_relu1')(conv9_bn1)
    conv9_conv2 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         name='dconv9_conv2')(conv9_relu1)
    conv9_bn2 = BatchNormalization(name='conv9_bn2')(conv9_conv2)
    # merge
    dconv9_merge = Multiply(name='dconv9_merge')([dconv9_bn, conv9_bn2])
    dconv9_out = ReLU(name='dconv9_out')(dconv9_merge)

    ## dconv10 # [None, 5, 5, 256]
    # Deconvolution layer
    dconv10_dconv = Convolution2DTranspose(filters=256,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same',
                                           output_padding=(0, 0),
                                           name='dconv10_dconv')(dconv9_out)
    dconv10_conv = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='dconv10_conv')(dconv10_dconv)
    dconv10_bn = BatchNormalization(name='dconv10_bn')(dconv10_conv)
    # Feature Layer from SSD
    conv10_conv1 = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='conv10_conv1')(conv6_block1_out)
    conv10_bn1 = BatchNormalization(name='conv10_bn1')(conv10_conv1)
    conv10_relu1 = ReLU(name='conv10_relu1')(conv10_bn1)
    conv10_conv2 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         name='dconv10_conv2')(conv10_relu1)
    conv10_bn2 = BatchNormalization(name='conv10_bn2')(conv10_conv2)
    # merge
    dconv10_merge = Multiply(name='dconv10_merge')([dconv10_bn, conv10_bn2])
    dconv10_out = ReLU(name='dconv10_out')(dconv10_merge)

    ## dconv11 # [None, 10, 10, 256]
    # Deconvolution layer
    dconv11_dconv = Convolution2DTranspose(filters=256,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same',
                                           name='dconv11_dconv')(dconv10_out)
    dconv11_conv = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='dconv11_conv')(dconv11_dconv)
    dconv11_bn = BatchNormalization(name='dconv11_bn')(dconv11_conv)
    # Feature Layer from SSD
    conv11_conv1 = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='conv11_conv1')(conv5_block3_out)
    conv11_bn1 = BatchNormalization(name='conv11_bn1')(conv11_conv1)
    conv11_relu1 = ReLU(name='conv11_relu1')(conv11_bn1)
    conv11_conv2 = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='dconv11_conv2')(conv11_relu1)
    conv11_bn2 = BatchNormalization(name='conv11_bn2')(conv11_conv2)
    # merge
    dconv11_merge = Multiply(name='dconv11_merge')([dconv11_bn, conv11_bn2])
    dconv11_out = ReLU(name='dconv11_out')(dconv11_merge)


    ## dconv12 # [None, 20, 20, 256]
    # Deconvolution layer
    dconv12_dconv = Convolution2DTranspose(filters=256,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same',
                                           name='dconv12_dconv')(dconv11_out)
    dconv12_conv = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='dconv12_conv')(dconv12_dconv)
    dconv12_bn = BatchNormalization(name='dconv12_bn')(dconv12_conv)
    # Feature Layer from SSD
    conv12_conv1 = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='conv12_conv1')(conv4_block23_out)
    conv12_bn1 = BatchNormalization(name='conv12_bn1')(conv12_conv1)
    conv12_relu1 = ReLU(name='conv12_relu1')(conv12_bn1)
    conv12_conv2 = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='dconv12_conv2')(conv12_relu1)
    conv12_bn2 = BatchNormalization(name='conv12_bn2')(conv12_conv2)
    # merge
    dconv12_merge = Multiply(name='dconv12_merge')([dconv12_bn, conv12_bn2])
    dconv12_out = ReLU(name='dconv12_out')(dconv12_merge)

    ## dconv13 # [None, 40, 40, 256]
    # Deconvolution layer
    dconv13_dconv = Convolution2DTranspose(filters=256,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same',
                                           name='dconv13_dconv')(dconv12_out)
    dconv13_conv = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='dconv13_conv')(dconv13_dconv)
    dconv13_bn = BatchNormalization(name='dconv13_bn')(dconv13_conv)
    # Feature Layer from SSD
    conv13_conv1 = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='conv13_conv1')(conv3_block4_out)
    conv13_bn1 = BatchNormalization(name='conv13_bn1')(conv13_conv1)
    conv13_relu1 = ReLU(name='conv13_relu1')(conv13_bn1)
    conv13_conv2 = Conv2D(filters=256,
                          kernel_size=(3, 3),
                          padding='same',
                          name='dconv13_conv2')(conv13_relu1)
    conv13_bn2 = BatchNormalization(name='conv13_bn2')(conv13_conv2)
    # merge
    dconv13_merge = Multiply(name='dconv13_merge')([dconv13_bn, conv13_bn2])
    dconv13_out = ReLU(name='dconv13_out')(dconv13_merge)


    # Predict Modul
    # conv8_block1_out
    predict8_block1_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict8_block1_0_conv')(conv8_block1_out)
    predict8_block1_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict8_block1_1_conv')(conv8_block1_out)
    predict8_block1_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict8_block1_2_conv')(predict8_block1_1_conv)
    predict8_block1_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict8_block1_3_conv')(predict8_block1_2_conv)
    predict8_block1_out = Add(name='predict8_block1_out')([predict8_block1_0_conv,predict8_block1_3_conv])
    predict8_block2_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict8_block2_0_conv')(predict8_block1_out)
    predict8_block2_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict8_block2_1_conv')(predict8_block1_out)
    predict8_block2_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict8_block2_2_conv')(predict8_block2_1_conv)
    predict8_block2_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict8_block2_3_conv')(predict8_block2_2_conv)
    predict8_block2_out = Add(name='predict8_block2_out')([predict8_block2_0_conv,predict8_block2_3_conv])

    # 處理anchor box
    num_priors = 4
    # num_priors表示每個網格點anchor box的數量，4是x,y,h,w的調整
    predict8_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='predict8_mbox_loc')(predict8_block2_out)
    predict8_mbox_loc_flat = Flatten(name='predict8_mbox_loc_flat')(predict8_mbox_loc)
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    predict8_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                name='predict8_mbox_conf')(predict8_mbox_loc)
    predict8_mbox_conf_flat = Flatten(name='predict8_mbox_conf_flat')(predict8_mbox_conf)
    predict8_mbox_priorbox = PriorBox(img_size, 264.0, max_size=315.0, aspect_ratios=[2],
                                     variances=[0.1, 0.1, 0.2, 0.2],
                                     name='predict8_mbox_priorbox')(predict8_block2_out)


    # dconv9_out
    predict9_block1_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict9_block1_0_conv')(dconv9_out)
    predict9_block1_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict9_block1_1_conv')(dconv9_out)
    predict9_block1_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict9_block1_2_conv')(predict9_block1_1_conv)
    predict9_block1_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict9_block1_3_conv')(predict9_block1_2_conv)
    predict9_block1_out = Add(name='predict9_block1_out')([predict9_block1_0_conv,predict9_block1_3_conv])
    predict9_block2_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict9_block2_0_conv')(predict9_block1_out)
    predict9_block2_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict9_block2_1_conv')(predict9_block1_out)
    predict9_block2_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict9_block2_2_conv')(predict9_block2_1_conv)
    predict9_block2_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict9_block2_3_conv')(predict9_block2_2_conv)
    predict9_block2_out = Add(name='predict9_block2_out')([predict9_block2_0_conv,predict9_block2_3_conv])

    # 處理anchor box
    num_priors = 4
    # num_priors表示每個網格點anchor box的數量，4是x,y,h,w的調整
    predict9_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='predict9_mbox_loc')(predict9_block2_out)
    predict9_mbox_loc_flat = Flatten(name='predict9_mbox_loc_flat')(predict9_mbox_loc)
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    predict9_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                name='predict9_mbox_conf')(predict9_mbox_loc)
    predict9_mbox_conf_flat = Flatten(name='predict9_mbox_conf_flat')(predict9_mbox_conf)
    predict9_mbox_priorbox = PriorBox(img_size, 213.0, max_size=264.0, aspect_ratios=[2],
                                      variances=[0.1, 0.1, 0.2, 0.2],
                                      name='predict9_mbox_priorbox')(predict9_block2_out)

    # dconv10_out
    predict10_block1_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict10_block1_0_conv')(dconv10_out)
    predict10_block1_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict10_block1_1_conv')(dconv10_out)
    predict10_block1_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict10_block1_2_conv')(predict10_block1_1_conv)
    predict10_block1_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict10_block1_3_conv')(predict10_block1_2_conv)
    predict10_block1_out = Add(name='predict10_block1_out')([predict10_block1_0_conv, predict10_block1_3_conv])
    predict10_block2_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict10_block2_0_conv')(predict10_block1_out)
    predict10_block2_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict10_block2_1_conv')(predict10_block1_out)
    predict10_block2_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict10_block2_2_conv')(predict10_block2_1_conv)
    predict10_block2_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict10_block2_3_conv')(predict10_block2_2_conv)
    predict10_block2_out = Add(name='predict10_block2_out')([predict10_block2_0_conv, predict10_block2_3_conv])


    # 處理anchor box
    num_priors = 6
    # num_priors表示每個網格點anchor box的數量，4是x,y,h,w的調整
    predict10_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='predict10_mbox_loc')(predict10_block2_out)
    predict10_mbox_loc_flat = Flatten(name='predict10_mbox_loc_flat')(predict10_mbox_loc)
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    predict10_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                name='predict10_mbox_conf')(predict10_mbox_loc)
    predict10_mbox_conf_flat = Flatten(name='predict10_mbox_conf_flat')(predict10_mbox_conf)
    predict10_mbox_priorbox = PriorBox(img_size, 162.0, max_size=213.0, aspect_ratios=[2, 3],
                                      variances=[0.1, 0.1, 0.2, 0.2],
                                      name='predict10_mbox_priorbox')(predict10_block2_out)

    # dconv11_out
    predict11_block1_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict11_block1_0_conv')(dconv11_out)
    predict11_block1_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict11_block1_1_conv')(dconv11_out)
    predict11_block1_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict11_block1_2_conv')(predict11_block1_1_conv)
    predict11_block1_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict11_block1_3_conv')(predict11_block1_2_conv)
    predict11_block1_out = Add(name='predict11_block1_out')([predict11_block1_0_conv, predict11_block1_3_conv])
    predict11_block2_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict11_block2_0_conv')(predict11_block1_out)
    predict11_block2_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict11_block2_1_conv')(predict11_block1_out)
    predict11_block2_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict11_block2_2_conv')(predict11_block2_1_conv)
    predict11_block2_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict11_block2_3_conv')(predict11_block2_2_conv)
    predict11_block2_out = Add(name='predict11_block2_out')([predict11_block2_0_conv, predict11_block2_3_conv])

    # 處理anchor box
    num_priors = 6
    # num_priors表示每個網格點anchor box的數量，4是x,y,h,w的調整
    predict11_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='predict11_mbox_loc')(predict11_block2_out)
    predict11_mbox_loc_flat = Flatten(name='predict11_mbox_loc_flat')(predict11_mbox_loc)
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    predict11_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='predict11_mbox_conf')(predict11_mbox_loc)
    predict11_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(predict11_mbox_conf)
    predict11_mbox_priorbox = PriorBox(img_size, 111.0, max_size=162.0, aspect_ratios=[2, 3],
                                      variances=[0.1, 0.1, 0.2, 0.2],
                                      name='predict11_mbox_priorbox')(predict11_block2_out)

    # dconv12_out
    predict12_block1_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict12_block1_0_conv')(dconv12_out)
    predict12_block1_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict12_block1_1_conv')(dconv12_out)
    predict12_block1_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict12_block1_2_conv')(predict12_block1_1_conv)
    predict12_block1_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict12_block1_3_conv')(predict12_block1_2_conv)
    predict12_block1_out = Add(name='predict12_block1_out')([predict12_block1_0_conv, predict12_block1_3_conv])
    predict12_block2_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict12_block2_0_conv')(predict12_block1_out)
    predict12_block2_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict12_block2_1_conv')(predict12_block1_out)
    predict12_block2_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict12_block2_2_conv')(predict12_block2_1_conv)
    predict12_block2_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict12_block2_3_conv')(predict12_block2_2_conv)
    predict12_block2_out = Add(name='predict12_block2_out')([predict12_block2_0_conv, predict12_block2_3_conv])

    # 處理anchor box
    num_priors = 6
    # num_priors表示每個網格點anchor box的數量，4是x,y,h,w的調整
    predict12_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='predict12_mbox_loc')(predict12_block2_out)
    predict12_mbox_loc_flat = Flatten(name='predict12_mbox_loc_flat')(predict12_mbox_loc)
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    predict12_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='predict12_mbox_conf')(predict12_mbox_loc)
    predict12_mbox_conf_flat = Flatten(name='predict12_mbox_conf_flat')(predict12_mbox_conf)
    predict12_mbox_priorbox = PriorBox(img_size, 60.0, max_size=111.0, aspect_ratios=[2, 3],
                                     variances=[0.1, 0.1, 0.2, 0.2],
                                     name='predict12_mbox_priorbox')(predict12_block2_out)


    # dconv13_out
    predict13_block1_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict13_block1_0_conv')(dconv13_out)
    predict13_block1_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict13_block1_1_conv')(dconv13_out)
    predict13_block1_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict13_block1_2_conv')(predict13_block1_1_conv)
    predict13_block1_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict13_block1_3_conv')(predict13_block1_2_conv)
    predict13_block1_out = Add(name='predict13_block1_out')([predict13_block1_0_conv, predict13_block1_3_conv])
    predict13_block2_0_conv = Conv2D(256, kernel_size=(1, 1), name='predict13_block2_0_conv')(predict13_block1_out)
    predict13_block2_1_conv = Conv2D(128, kernel_size=(1, 1), name='predict13_block2_1_conv')(predict13_block1_out)
    predict13_block2_2_conv = Conv2D(128, kernel_size=(1, 1), name='predict13_block2_2_conv')(predict13_block2_1_conv)
    predict13_block2_3_conv = Conv2D(256, kernel_size=(1, 1), name='predict13_block2_3_conv')(predict13_block2_2_conv)
    predict13_block2_out = Add(name='predict13_block2_out')([predict13_block2_0_conv, predict13_block2_3_conv])

    # 處理anchor box
    num_priors = 4
    # num_priors表示每個網格點anchor box的數量，4是x,y,h,w的調整
    predict13_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='predict13_mbox_loc')(predict13_block2_out)
    predict13_mbox_loc_flat = Flatten(name='predict13_mbox_loc_flat')(predict13_mbox_loc)
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    predict13_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same', name='predict13_mbox_conf')(predict13_mbox_loc)
    predict13_mbox_conf_flat = Flatten(name='predict13_mbox_conf_flat')(predict13_mbox_conf)
    predict13_mbox_priorbox = PriorBox(img_size, 30.0,max_size = 60.0, aspect_ratios=[2],
                                      variances=[0.1, 0.1, 0.2, 0.2],
                                      name='predict13_mbox_priorbox')(predict13_block2_out)


    # 将所有结果进行堆叠
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([predict8_mbox_loc_flat,
                                                     predict9_mbox_loc_flat,
                                                     predict10_mbox_loc_flat,
                                                     predict11_mbox_loc_flat,
                                                     predict12_mbox_loc_flat,
                                                     predict13_mbox_loc_flat])
                            
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([predict8_mbox_conf_flat,
                                                       predict9_mbox_conf_flat,
                                                       predict10_mbox_conf_flat,
                                                       predict11_mbox_conf_flat,
                                                       predict12_mbox_conf_flat,
                                                       predict13_mbox_conf_flat])
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([predict8_mbox_priorbox,
                                                               predict9_mbox_priorbox,
                                                               predict10_mbox_priorbox,
                                                               predict11_mbox_priorbox,
                                                               predict12_mbox_priorbox,
                                                               predict13_mbox_priorbox])
                                  
    # 8732,4
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)
    # 8732,21
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions')([mbox_loc,
                                                           mbox_conf,
                                                           mbox_priorbox])


    model = Model(inputs=input_tensor, outputs=predictions)
    model.summary()
    return model