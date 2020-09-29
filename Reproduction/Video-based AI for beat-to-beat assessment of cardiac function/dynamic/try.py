import echonet
echonet.utils.segmentation.run(modelname="deeplabv3_resnet50",save_segmentation=True,pretrained=False)
echonet.utils.video.run(modelname="r2plus1d_18",
                                             frames=32,
                                             period=2,
                                             pretrained=True,
                                             batch_size=8)