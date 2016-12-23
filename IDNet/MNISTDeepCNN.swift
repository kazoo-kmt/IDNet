//
//  MNISTDeepCNN.swift
//  IDNet
//
//  Created by Kazu Komoto on 12/19/16.
//  Copyright © 2016 Kazu Komoto. All rights reserved.
//
//
/*
 Deep layer network where we define and encode the correct layers on a command buffer as needed
 This is based on MNISTSingleLayer.swift and MNISTDeepCNN.swift provided by Apple,
 and MNISTDeepCNN.swift provided by Shuichi Tsutsumi.
 */

import MetalPerformanceShaders
import Accelerate


class MNISTDeepCNN {

  // MPSImageDescriptors for different layers outputs to be put in
  let sid  = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 8, height: 200, featureChannels: 1)  // img_rows = 8, img_cols = 200
  let c1id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 8, height: 200, featureChannels: 20)
  let c2id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 5, height: 191, featureChannels: 40)
  let pid  = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 1, height: 63, featureChannels: 40)
  let fc1id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 1 , height: 1 , featureChannels: 40)
  let did  = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 1, height: 1, featureChannels: 50)
  
  // MPSImages and layers declared
  var srcImage, dstImage : MPSImage
  var c1Image, c2Image, pImage, fc1Image: MPSImage
  var conv1, conv2: MPSCNNConvolution
  var fc1, fc2: MPSCNNFullyConnected
  var pool: MPSCNNPoolingMax
  var relu: MPSCNNNeuronReLU
  var tanh: MPSCNNNeuronTanH
  var softmax : MPSCNNSoftMax
//  var softmax : MPSCNNNeuronTanH
  
  var commandQueue : MTLCommandQueue
  var device : MTLDevice
  
  init(withCommandQueue commandQueueIn: MTLCommandQueue!) {
    commandQueue = commandQueueIn
    device = commandQueueIn.device
    
    pool = MPSCNNPoolingMax(device: device, kernelWidth: 3, kernelHeight: 3, strideInPixelsX: 3, strideInPixelsY: 3)
//    pool.offset = MPSOffset(x: 1, y: 1, z: 0)
//    pool.edgeMode = MPSImageEdgeMode.clamp
    relu = MPSCNNNeuronReLU(device: device, a: 0)
    tanh = MPSCNNNeuronTanH(device: device, a: 1, b: 1)
    
    
    // Initialize MPSImage from descriptors
    c1Image     = MPSImage(device: device, imageDescriptor: c1id)
    c2Image     = MPSImage(device: device, imageDescriptor: c2id)
    pImage      = MPSImage(device: device, imageDescriptor: pid)
    fc1Image    = MPSImage(device: device, imageDescriptor: fc1id)
    
    
    // setup convolution layers
    conv1 = SlimMPSCNNConvolution(kernelWidth: 2,
                                  kernelHeight: 10,
                                  inputFeatureChannels: 1,
                                  outputFeatureChannels: 20,
                                  neuronFilter: relu,
                                  device: device,
                                  kernelParamsBinaryName: "conv1")
    
    conv2 = SlimMPSCNNConvolution(kernelWidth: 4,
                                  kernelHeight: 10,
                                  inputFeatureChannels: 20,
                                  outputFeatureChannels: 40,
                                  neuronFilter: tanh,
                                  device: device,
                                  kernelParamsBinaryName: "conv2")
    
    fc1 = SlimMPSCNNFullyConnected(kernelWidth: 1,
                                   kernelHeight: 63,
                                   inputFeatureChannels: 40,
                                   outputFeatureChannels: 40,
                                   neuronFilter: tanh,
                                   device: device,
                                   kernelParamsBinaryName: "fc1")
    
    fc2 = SlimMPSCNNFullyConnected(kernelWidth: 1,
                                   kernelHeight: 1,
                                   inputFeatureChannels: 40,
                                   outputFeatureChannels: 50,
                                   neuronFilter: nil,
                                   device: device,
                                   kernelParamsBinaryName: "fc2")
    
    // Initialize MPSImage from descriptors
    srcImage = MPSImage(device: device, imageDescriptor: sid)
    dstImage = MPSImage(device: device, imageDescriptor: did)
    
    // prepare softmax layer to be applied at the end to get a clear label
    softmax = MPSCNNSoftMax(device: device)
//    softmax = MPSCNNNeuronSigmoid(device: device)

  }
  
  
  /**
   This function encodes all the layers of the network into given commandBuffer, it calls subroutines for each piece of the network
   
   - Parameters:
   - inputImage: Image coming in on which the network will run
   - imageNum: If the test set is being used we will get a value between 0 and 9999 for which of the 10,000 images is being evaluated
   - correctLabel: The correct label for the inputImage while testing
   
   - Returns:
   Guess of the network as to what the digit is as UInt
   */
  func forward(inputImage: MPSImage? = nil, imageNum: Int = 9999, correctLabel: UInt = 50) -> UInt{
    var label = UInt(99)
    
    // to deliver optimal performance we leave some resources used in MPSCNN to be released at next call of autoreleasepool,
    // so the user can decide the appropriate time to release this
    autoreleasepool{
      // Get command buffer to use in MetalPerformanceShaders.
      let commandBuffer = commandQueue.makeCommandBuffer()
      
      // output will be stored in this image
      let finalLayer = MPSImage(device: commandBuffer.device, imageDescriptor: did)
      
      
      
//      var outputSrc = [UInt16](repeating: 0, count: 1600)
//      let regionSrc = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
//                                  size: MTLSize(width: 8, height: 200, depth: 1))
//      for i in 0..<1 {
//        srcImage.texture.getBytes(&(outputSrc[8 * 4 * i]),
//                                  bytesPerRow: 8 * MemoryLayout<UInt16>.size,
//                                  bytesPerImage: 0,
//                                  from: regionSrc,
//                                  mipmapLevel: 0,
//                                  slice: i)
//      }
//      
      
      
      // encode layers to metal commandBuffer
      if inputImage == nil {
        conv1.encode(commandBuffer: commandBuffer, sourceImage: srcImage, destinationImage: c1Image)
      }
      else{
        conv1.encode(commandBuffer: commandBuffer, sourceImage: inputImage!, destinationImage: c1Image)
      }
      

      
//      var outputConv1 = [UInt16](repeating: 0, count: 8*200*20)
//      let regionConv1 = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
//                                  size: MTLSize(width: 8, height: 200, depth: 1))
//      for i in 0..<5 {
//        c1Image.texture.getBytes(&(outputConv1[8 * 4 * i]),
//                                  bytesPerRow: 8 * MemoryLayout<UInt16>.size * 4,
//                                  bytesPerImage: 0,
//                                  from: regionConv1,
//                                  mipmapLevel: 0,
//                                  slice: i)
//      }
      
      
      
      conv2.encode  (commandBuffer: commandBuffer, sourceImage: c1Image   , destinationImage: c2Image)
   
      
      
//      var outputConv2 = [UInt16](repeating: 0, count: 5*191*40)
//      let regionConv2 = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
//                                  size: MTLSize(width: 5, height: 191, depth: 1))
//      for i in 0..<10 {
//        c2Image.texture.getBytes(&(outputConv2[5 * 4 * i]),
//                                  bytesPerRow: 5 * MemoryLayout<UInt16>.size * 4,
//                                  bytesPerImage: 0,
//                                  from: regionConv2,
//                                  mipmapLevel: 0,
//                                  slice: i)
//      }
      
      
      
      pool.encode   (commandBuffer: commandBuffer, sourceImage: c2Image   , destinationImage: pImage)
      
      
//      
//      var outputPool = [UInt16](repeating: 0, count: 1*63*40)
//      let regionPool = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
//                                  size: MTLSize(width: 1, height: 63, depth: 1))
//      for i in 0..<10 {
//        pImage.texture.getBytes(&(outputPool[1 * 4 * i]),
//                                  bytesPerRow: 1 * MemoryLayout<UInt16>.size * 4,
//                                  bytesPerImage: 0,
//                                  from: regionPool,
//                                  mipmapLevel: 0,
//                                  slice: i)
//      }
//      
    
      
      fc1.encode    (commandBuffer: commandBuffer, sourceImage: pImage   , destinationImage: fc1Image)
      
//      
//      
//      var outputFc1 = [UInt16](repeating: 0, count: 1*1*40)
//      let regionFc1 = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
//                                 size: MTLSize(width: 1, height: 1, depth: 1))
//      for i in 0..<10 {
//        fc1Image.texture.getBytes(&(outputFc1[1 * 4 * i]),
//                                  bytesPerRow: 1 * MemoryLayout<UInt16>.size * 4,
//                                  bytesPerImage: 0,
//                                  from: regionFc1,
//                                  mipmapLevel: 0,
//                                  slice: i)
//      }
//      
//  
      
      fc2.encode    (commandBuffer: commandBuffer, sourceImage: fc1Image  , destinationImage: dstImage)
      
      
//      
//      var outputDst = [UInt16](repeating: 0, count: 1*1*50)
//      let regionDst = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
//                                size: MTLSize(width: 1, height: 1, depth: 1))
//      for i in 0..<13 {
//        dstImage.texture.getBytes(&(outputDst[1 * 4 * i]),
//                                  bytesPerRow: 1 * MemoryLayout<UInt16>.size * 4,
//                                  bytesPerImage: 0,
//                                  from: regionDst,
//                                  mipmapLevel: 0,
//                                  slice: i)
//      }
//      
//      
      
      softmax.encode(commandBuffer: commandBuffer, sourceImage: dstImage  , destinationImage: finalLayer)
      
      // add a completion handler to get the correct label the moment GPU is done and compare it to the correct output or return it
//      var arrayConv1 = [UInt16](repeating: 0, count: 8*200*40)
      var arrayConv2 = [UInt16](repeating: 0, count: 5*191*40)
      commandBuffer.addCompletedHandler { commandBuffer in
        label = self.getLabel(finalLayer: finalLayer)
//        arrayConv1 = self.getImageConv1(c1Image: self.c1Image)
        arrayConv2 = self.getImageConv2(c2Image: self.c2Image)  //NTR: The output of conv2 is weired.
      }
      
      // commit commandbuffer to run on GPU and wait for completion
      commandBuffer.commit()
      if imageNum == 9999 {
        commandBuffer.waitUntilCompleted()
      }
      
    }
    return label
  }
  
  /**
   This function reads the output probabilities from finalLayer to CPU, sorts them and gets the label with heighest probability
   
   - Parameters:
   - finalLayer: output image of the network this has probabilities of each digit
   
   - Returns:
   Guess of the network as to what the digit is as UInt
   */
  
  
  
  
//NTR: Need to fix below.
  
  
  
  
  func getLabel(finalLayer: MPSImage) -> UInt {
    // even though we have 50 labels outputed the MTLTexture format used is RGBAFloat16 thus 17 slices will have 13*4 = 52 outputs
    var result_half_array = [UInt16](repeating: 6, count: 52)
    var result_float_array = [Float](repeating: 0.3, count: 50)
    for i in 0..<13 {
      finalLayer.texture.getBytes(&(result_half_array[4*i]),
                                  bytesPerRow: MemoryLayout<UInt16>.size*1*4,
                                  bytesPerImage: MemoryLayout<UInt16>.size*1*1*4,
                                  from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                                  size: MTLSize(width: 1, height: 1, depth: 1)),
                                  mipmapLevel: 0,
                                  slice: i)
    }
    
    // we use vImage to convert our data to float16, Metal GPUs use float16 and swift float is 32-bit
    var fullResultVImagebuf = vImage_Buffer(data: &result_float_array, height: 1, width: 50, rowBytes: 50*4)
    var halfResultVImagebuf = vImage_Buffer(data: &result_half_array , height: 1, width: 50, rowBytes: 50*2)
    
    if vImageConvert_Planar16FtoPlanarF(&halfResultVImagebuf, &fullResultVImagebuf, 0) != kvImageNoError {
      print("Error in vImage")
    }
    
    // poll all labels for probability and choose the one with max probability to return
    var max:Float = 0
//    var mostProbableDigit = 10
    var mostProbableDigit = 50
//    for i in 0...9 {
    for i in 0..<50 {
      if(max < result_float_array[i]){
        max = result_float_array[i]
        mostProbableDigit = i
      }
    }
    
    return UInt(mostProbableDigit)
  }
  
  func getImageConv1(c1Image: MPSImage) -> [UInt16] {
    var outputConv1 = [UInt16](repeating: 0, count: 8*200*20)
    let regionConv1 = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                size: MTLSize(width: 8, height: 200, depth: 1))
    for i in 0..<5 {
      c1Image.texture.getBytes(&(outputConv1[8 * 4 * i]),
                               bytesPerRow: 8 * MemoryLayout<UInt16>.size * 4,
                               bytesPerImage: 0,
                               from: regionConv1,
                               mipmapLevel: 0,
                               slice: i)
    }
    
    return outputConv1
  }
  
  func getImageConv2(c2Image: MPSImage) -> [UInt16] {
    var outputConv2 = [UInt16](repeating: 0, count: 5*191*40)
    let regionConv2 = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                size: MTLSize(width: 5, height: 191, depth: 1))
    for i in 0..<10 {
      c2Image.texture.getBytes(&(outputConv2[5 * 4 * i]),
                               bytesPerRow: 5 * MemoryLayout<UInt16>.size * 4,
                               bytesPerImage: 0,
                               from: regionConv2,
                               mipmapLevel: 0,
                               slice: i)
    }
    
    return outputConv2
  }
}
