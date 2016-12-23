//
//  ViewController.swift
//  IDNet
//
//  Created by Kazu Komoto on 12/19/16.
//  Copyright © 2016 mycompany. All rights reserved.
//

import UIKit
import MetalPerformanceShaders

class ViewController: UIViewController {
  
  var commandQueue: MTLCommandQueue!
  var device: MTLDevice!
  
  // Networks we have
  var network: MNISTDeepCNN!
  
  // dataset image parameters
  let inputWidth  = 8
  let inputHeight = 200
  
  @IBOutlet private weak var predictionLabel: UILabel!
  


  override func viewDidLoad() {
    super.viewDidLoad()
    // Do any additional setup after loading the view, typically from a nib.
    
    let csvFloatArray = CsvToArray.csvToFloatArray(csvFileName: "test_0")
//    print(csvFloatArray)
//    let dummyFloatArray = [Float](repeating: 0.0, count: 8*200)
    
    predictionLabel.text = nil
    
    // Load default device.
    device = MTLCreateSystemDefaultDevice()
    
    // Make sure the current device supports MetalPerformanceShaders.
    guard MPSSupportsMTLDevice(device) else {
      showAlert(title: "Not Supported", message: "MetalPerformanceShaders is not supported on current device", handler: { (action) in
        self.navigationController!.popViewController(animated: true)
      })
      return
    }
    
    // Create new command queue.
    commandQueue = device!.makeCommandQueue()
    
    // initialize the networks we shall use to detect digits
    network  = MNISTDeepCNN(withCommandQueue: commandQueue)
    
    
    // validate NeuralNetwork was initialized properly
    assert(network != nil)
    
    
    // putting input into MTLTexture in the MPSImage
//    var csvData = NSData(bytes: csvFloatArray, length: inputWidth * inputHeight * MemoryLayout<Float>.size)
//    print(csvData)
    
    //    // putting input into MTLTexture in the MPSImage
    //    network.srcImage.texture.replace(region: MTLRegion( origin: MTLOrigin(x: 0, y: 0, z: 0),
    //                                                        size: MTLSize(width: mnistInputWidth, height: mnistInputHeight, depth: 1)),
    //                                     mipmapLevel: 0,
    //                                     slice: 0,
    //                                     withBytes: context!.data!,
    //                                     bytesPerRow: mnistInputWidth,
    //                                     bytesPerImage: 0)
    // putting input into MTLTexture in the MPSImage

    network.srcImage.texture.replace(region: MTLRegion( origin: MTLOrigin(x: 0, y: 0, z: 0),
                                                        size: MTLSize(width: inputWidth, height: inputHeight, depth: 1)),
                                     mipmapLevel: 0,
                                     slice: 0,
                                     withBytes: csvFloatArray,
//                                     bytesPerRow: 0,
                                     bytesPerRow: inputWidth * MemoryLayout<Float>.size,  //NTR: If this is just inputWidth, it returns an assertion error.
                                     bytesPerImage: 0)

    // run the network forward pass
    let label = network.forward()
    
    // show the prediction
    predictionLabel.text = "\(label)"
    
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }


}

