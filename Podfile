# Uncomment this line to define a global platform for your project
# platform :ios, '10.0'
source 'https://github.com/CocoaPods/Specs.git'

target 'IDNet' do
  # Comment this line if you're not using Swift and don't want to use dynamic frameworks
  use_frameworks!

  # Pods for IDNet
  pod 'HDF5Kit', :git => 'https://github.com/aleph7/HDF5Kit.git'

end

post_install do |installer|
    installer.pods_project.targets.each do |target|
        target.build_configurations.each do |config|
            config.build_settings['SWIFT_VERSION'] = '3.0'
        end
    end
end