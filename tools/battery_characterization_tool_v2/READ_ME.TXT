First, the .toml configuration file for your battery_vendor is required to exist in the battery_models directory, 
so you can use it to load the battery_formatting
The neccesary variables are:
[
dataset_path
json_path
battery_manufacturer
batteries_to_process
temperatures_to_process
]

Set the config file name to {vendor_name}.toml and make sure the last part of the dataset for this battery in the dataset directory ends in _{vendor_name}
For example, for a battery_vendor ABCD, 
a config file in the battery_models directory would look like: ABCD.toml 
and the dataset folder for the data would be named something like: battery_data_ABCD   

=======
MAKE SURE THE NAME OF THE CONFIG FILE, THE BATTERY_VENDOR VARIABLE AND THE ENDING OF THE DATASET FOLDER ARE ALL IDENTICAL AND NAMED AFTER THE CORECT VENDOR
=======

Start the battery_formatting python program and choose the correct configuration file (with the right dataset specified in the .toml!)

A JSON of the data is created in the exported_libraries_updated/battery_models_json named: {vendor}_{dataset_hash}.json_path

To create a different formatting using a different dataset, simply change the dataset_path to the different dataset folder and a new output json is created

Now run battery_simulation and choose the right datset you want to analyze in the console, 
then CHOOSE THE .JSON FILE WHICH WAS CREATED FROM THE SAME DATASET PREVIOUSLY USING battery_formatting 

All .c and .h files area created in the exported_libraries_updated folder 
All plots are stored in regression_results in the folder named: {vendor}_{dataset_hash}