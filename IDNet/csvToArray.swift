//
//  csvToArray.swift
//  IDNet
//
//  Created by Kazu Komoto on 12/20/16.
//  Copyright © 2016 mycompany. All rights reserved.
//

import Foundation
class CsvToArray {
  
  static func csvToFloatArrayOfArrays(csvFileName: String) -> [[Float]] {
    var result: [[String]] = []
    if let csvPath = Bundle.main.path(forResource: csvFileName, ofType: "csv") {
      guard let csvString = try? NSString(contentsOfFile: csvPath, encoding: String.Encoding.utf8.rawValue) else {
        fatalError("Failed to get strings from csv file")
      }
      csvString.enumerateLines { (line, stop) -> () in
        result.append(line.components(separatedBy: ","))
      }
    }
    let floatResult = result.map { $0.map {Float($0) ?? 0.0}}
    
    return floatResult
  }
  
  static func csvToFloatArray(csvFileName: String) -> [Float] {
    var result: [[String]] = []
    if let csvPath = Bundle.main.path(forResource: csvFileName, ofType: "csv") {
      guard let csvString = try? NSString(contentsOfFile: csvPath, encoding: String.Encoding.utf8.rawValue) else {
        fatalError("Failed to get strings from csv file")
      }
      csvString.enumerateLines { (line, stop) -> () in
        result.append(line.components(separatedBy: ","))
      }
    }
    let floatResult = result.map { $0.map {Float($0) ?? 0.0}}
    
    return floatResult.reduce([], +)
  }




}
