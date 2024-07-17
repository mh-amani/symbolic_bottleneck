import importlib

def instantiate_from_config(config):
    # Extract the class path and the parameters from the config
    class_path = config['_target_']
    module_name, class_name = class_path.rsplit('.', 1)

    # Dynamically import the module and get the class
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Instantiate the class with the remaining config parameters
    if 'pretrained_model_name' in config:
        pretrained_model_name = config['pretrained_model_name']
        # Instantiate the pre-trained model
        instance = cls.from_pretrained(pretrained_model_name, config.get('config', None))
    else:
        # Instantiate the class with the remaining config parameters    
        if config.get('config', None):
            instance = cls(config['config'])
        else:
            if config.get('_target_', None):
                config.pop('_target_')
            instance = cls(**config)
            
    return instance
