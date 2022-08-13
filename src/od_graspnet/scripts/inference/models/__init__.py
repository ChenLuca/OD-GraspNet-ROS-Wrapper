def get_network(network_name):
    network_name = network_name.lower()  
   
    if network_name == "odc_shuffle_v2_4":
        from .ODC_Shuffle_v2_ConvNet_4 import Generative_ODC_Shuffle_v2_4

    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
