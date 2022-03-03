DUAL_IN_A3C_MODEL_CFG = {
    "SimpleModel": {
        "ego_layers_structure": ["FC_64", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_8|K3|S2", "A_ReLU", "C1_16|K3|S2", "A_ReLU", "C1_32|K3|S2", "A_ReLU", "F"],
        "shared_layers_structure": [],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1"],
    },
    "VGG_MiniA": {
        "ego_layers_structure": ["FC_64", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_16|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_32|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_64|K3|P1", "A_ReLU", "C1_64|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_128|K3|P1", "A_ReLU", "C1_128|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_128|K3|P1", "A_ReLU", "C1_128|K3|P1", "A_ReLU", "M1_K2", "F"],
        "shared_layers_structure": [],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1"]
    },
    "VGG_MiniA_SigmoidValue": {
        "ego_layers_structure": ["FC_64", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_16|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_32|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_64|K3|P1", "A_ReLU", "C1_64|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_128|K3|P1", "A_ReLU", "C1_128|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_128|K3|P1", "A_ReLU", "C1_128|K3|P1", "A_ReLU", "M1_K2", "F"],
        "shared_layers_structure": [],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VGG_MiniA_SigmoidValue_With_More_Shared_Layers": {
        "ego_layers_structure": ["FC_64", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_16|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_32|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_64|K3|P1", "A_ReLU", "C1_64|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_128|K3|P1", "A_ReLU", "C1_128|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_128|K3|P1", "A_ReLU", "C1_128|K3|P1", "A_ReLU", "M1_K2", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VGG_MiniA_SigmoidValue_With_More_Shared_Layers_And_Larger_Fully_Connected_Ego_Layer": {
        "ego_layers_structure": ["FC_128", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_16|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_32|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_64|K3|P1", "A_ReLU", "C1_64|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_128|K3|P1", "A_ReLU", "C1_128|K3|P1", "A_ReLU", "M1_K2",
                                    "C1_128|K3|P1", "A_ReLU", "C1_128|K3|P1", "A_ReLU", "M1_K2", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VGG_2D_SigmoidValue_With_More_Shared_Layers": {  # Like VGG but with 2D spatial input (3 lanes)
        "ego_layers_structure": ["FC_64", "A_ReLU", "F"],
        "actors_layers_structure": ["C2_16|K(1,3)|P(0,1)", "A_ReLU", "M2_K(1,2)",
                                    "C2_32|K(1,3)|P(0,1)", "A_ReLU", "M2_K(1,2)",
                                    "C2_64|K(1,3)|P(0,1)", "A_ReLU",
                                    "C2_64|K3|P1", "A_ReLU", "M2_K(1,2)",
                                    "C2_128|K(1,3)|P(0,1)", "A_ReLU",
                                    "C2_128|K(1,3)|P(0,1)", "A_ReLU", "M2_K(1,2)",
                                    "C2_128|K(1,3)|P(0,1)", "A_ReLU",
                                    "C2_128|K(1,3)|P(0,1)", "A_ReLU", "M2_K(1,2)",
                                    "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VGG_2D_SigmoidValue_With_Double_Shared_Layers": {  # Like former, with more shared layers
        "ego_layers_structure": ["FC_64", "A_ReLU", "F"],
        "actors_layers_structure": ["C2_16|K(1,3)|P(0,1)", "A_ReLU", "M2_K(1,2)",
                                    "C2_32|K(1,3)|P(0,1)", "A_ReLU", "M2_K(1,2)",
                                    "C2_64|K(1,3)|P(0,1)", "A_ReLU",
                                    "C2_64|K3|P1", "A_ReLU", "M2_K(1,2)",
                                    "C2_128|K(1,3)|P(0,1)", "A_ReLU",
                                    "C2_128|K(1,3)|P(0,1)", "A_ReLU", "M2_K(1,2)",
                                    "C2_128|K(1,3)|P(0,1)", "A_ReLU",
                                    "C2_128|K(1,3)|P(0,1)", "A_ReLU", "M2_K(1,2)",
                                    "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },

    "ALPHA_SHALLOW_V1": {  # Like https://www.nature.com/articles/nature24270.epdf - less blocks and channels, no BN
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_32|K3", "RB1_32|K3|P1", "RB1_32|K3|P1", "RB1_32|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_SHALLOW_V1_Slim": {  # Like V1 but with 16 channels for ResBlocks
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_16|K3", "RB1_16|K3|P1", "RB1_16|K3|P1", "RB1_16|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_SHALLOW_V1_Fat": {  # Like V1 but with 16 channels for ResBlocks
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_64|K3", "RB1_64|K3|P1", "RB1_64|K3|P1", "RB1_64|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_SHALLOWER_V1": {  # SHALLOWER - Like V1 but x2 deep instead of x3
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_32|K3", "RB1_32|K3|P1", "RB1_32|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_SHALLOWER_V1_Slim": {  # Like V1_Slim but only 2 res-blocks
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_16|K3", "RB1_16|K3|P1", "RB1_16|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_SHALLOWER_V1_Fat": {  # SHALLOWER (x2) with 64 channels
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_64|K3", "RB1_64|K3|P1", "RB1_64|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_DEEPER_V1": {  # DEEPER (x4) with 32 channels
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_32|K3", "RB1_32|K3|P1", "RB1_32|K3|P1", "RB1_32|K3|P1", "RB1_32|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_DEEPER_V1_ActorEmb": {  # DEEPER (x4) with 32 channels
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_32|K3", "RB1_32|K3|P1", "RB1_32|K3|P1", "RB1_32|K3|P1", "RB1_32|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_DEEPER_V1_ActorEmb_FatShared": {  # DEEPER (x4) with 32 channels
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_32|K3", "RB1_32|K3|P1", "RB1_32|K3|P1", "RB1_32|K3|P1", "RB1_32|K3|P1", "F"],
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_DEEPER_V1_Slim": {  # DEEPER (x4) with 16 channels
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_16|K3", "RB1_16|K3|P1", "RB1_16|K3|P1", "RB1_16|K3|P1", "RB1_16|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_DEEPER_V1_Fat": {  # DEEPER (x64) with 16 channels
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_64|K3", "RB1_64|K3|P1", "RB1_64|K3|P1", "RB1_64|K3|P1", "RB1_64|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_BIGGER_ActorEmb": {  # DEEPER (x64) with 16 channels
        "ego_layers_structure": ["F", "FC_64", "A_ReLU"],
        "actors_layers_structure": ["C1_64|K3", "RB1_64|K3|P1", "RB1_64|K3|P1", "RB1_64|K3|P1", "RB1_64|K3|P1", "F"],
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_BIGGER_ActorEmb_FatShared": {  # DEEPER (x64) with 16 channels
        "ego_layers_structure": ["F", "FC_64", "A_ReLU"],
        "actors_layers_structure": ["C1_64|K3", "RB1_64|K3|P1", "RB1_64|K3|P1", "RB1_64|K3|P1", "RB1_64|K3|P1", "F"],
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_2D_V1": {
        "ego_layers_structure": ["F", "FC_64", "A_ReLU"],
        "actors_layers_structure": ["C2_32|K(1,3)", "A_ReLU", "RB2_32|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "C2_64|K(1,3)", "A_ReLU", "RB2_64|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "F"],
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_2D_V2": {
        "ego_layers_structure": ["F", "FC_64", "A_ReLU"],
        "actors_layers_structure": ["C2_32|K(1,3)", "A_ReLU", "RB2_32|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "C2_64|K(1,3)", "A_ReLU", "RB2_64|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "C2_128|K(1,3)", "A_ReLU", "RB2_128|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "F"],
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "ALPHA_2D_V3": {
        "ego_layers_structure": ["F", "FC_64", "A_ReLU"],
        "actors_layers_structure": ["C2_32|K(1,3)", "A_ReLU", "RB2_32|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "C2_64|K(1,3)", "A_ReLU", "RB2_64|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "C2_128|K(1,3)", "A_ReLU", "RB2_128|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "C2_128|K(1,3)", "A_ReLU", "RB2_128|K(1,3)|P(0,1)", "M2_K(1,2)",
                                    "F"],
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V1": {  # As in https://arxiv.org/abs/1905.02680
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_32|K1|S1", "A_ReLU", "C1_32|K1", "A_ReLU", "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_64", "A_ReLU", "FC_64", "A_ReLU"],
        "actor_head_structure": ["FC_32", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_32", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V1_HP": {  # As in https://arxiv.org/abs/1905.02680
        "ego_layers_structure": ["F", "FC_64", "A_ReLU"],
        "actors_layers_structure": ["C1_32|K1|S1", "A_ReLU", "C1_32|K1", "A_ReLU", "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_64", "A_ReLU", "FC_64", "A_ReLU"],
        "actor_head_structure": ["FC_32", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_32", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V2": {  # Like VOLVONET_V1 but Slimmer
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_16|K1|S1", "A_ReLU", "C1_16|K1", "A_ReLU", "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_64", "A_ReLU", "FC_64", "A_ReLU"],
        "actor_head_structure": ["FC_32", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_32", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V3": {  # Like VOLVONET_V1 but Wider
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "C1_64|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V3_HP": {  # Like VOLVONET_V1 but Wider
        "ego_layers_structure": ["F", "FC_64", "A_ReLU"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "C1_64|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V4": {  # Like VOLVONET_V1 but Wider
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V5": {  # Like VOLVONET_V1 but Deeper and Wider
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "C1_64|K1", "A_ReLU",
                                    "C1_64|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V5_HP": {  # Like VOLVONET_V1 but Deeper and Wider
        "ego_layers_structure": ["F", "FC_64", "A_ReLU"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "C1_64|K1", "A_ReLU",
                                    "C1_64|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V6": {  # Like VOLVONET_V1 but Deeper and Wider
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU", "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V7": {  # Like VOLVONET_V1 but Deeper and Wider
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V7_HP": {  # Like VOLVONET_V1 but Deeper and Wider
        "ego_layers_structure": ["C1_64|K1", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V6_TH": {  # Like VOLVONET_V6 but with Tanh activation on heads
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_Tanh", "FC_128", "A_Tanh"],
        "actor_head_structure": ["FC_128", "A_Tanh", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_Tanh", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V7_TH": {  # Like VOLVONET_V7 but with Tanh activation on heads
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_Tanh",
                                    "FC_256", "A_Tanh"],
        "actor_head_structure": ["FC_128", "A_Tanh",
                                 "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_Tanh",
                                  "FC_1", "A_Sigmoid"]
    },

    "VOLVONET_V5_RB": {  # Like VOLVONET_V5 with ResBlock based actors module
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_64|K1",
                                    "RB1_64|K1|P0", "A_ReLU",
                                    "RB1_64|K1|P0", "A_ReLU",
                                    "RB1_64|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU",
                                    "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU",
                                 "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU",
                                  "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V5_RB_HP": {  # Like VOLVONET_V5_ResBlocks, with additional ResBlock for host
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_64|K1",
                                    "RB1_64|K1|P0", "A_ReLU",
                                    "RB1_64|K1|P0", "A_ReLU",
                                    "RB1_64|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU",
                                    "FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU",
                                 "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU",
                                  "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V5_RB_HP_Shallow": {  # Like VOLVONET_V5_ResBlocks, with additional ResBlock for host
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_64|K1",
                                    "RB1_64|K1|P0", "A_ReLU",
                                    "RB1_64|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_128", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V7_RB": {  # Like VOLVONET_V7 with ResBlock based actors module
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V7_RB_HP": {  # Like VOLVONET_V7 with ResBlock based actors module
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_128|K1",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V7_RB_HP_Shallow": {  # Like VOLVONET_V7 with ResBlock based actors module
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_128|K1",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V7_RB_HP_Shallow_FatHeads": {  # Like VOLVONET_V7 with ResBlock based actors module
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_128|K1",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V8": {  # Like VOLVONET_V7_RB_HP_Shallow with double backbone layers
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_256|K1",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V8_FatHeads": {  # Like VOLVONET_V7_RB_HP_Shallow_FatHeads with double backbone layers
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_256|K1",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V8_FatHeads2": {  # Like VOLVONET_V7_RB_HP_Shallow_FatHeads with double backbone layers
        "ego_layers_structure": ["C1_64|K1", "A_ReLU",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_256|K1", "A_ReLU",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_512", "A_ReLU", "FC_512", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "VOLVONET_V9": {  # Like VOLVONET_V7_RB_HP_Shallow_FatHeads with double backbone layers
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_256|K1",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "RB1_256|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_512", "A_ReLU", "FC_512", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },


    "VOLVONET_V7_TanhCritic": {  # Like VOLVONET_V1 but Deeper and Wider
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "C1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "VOLVONET_V7_RB_TanhCritic": {  # Like VOLVONET_V7 with ResBlock based actors module
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "VOLVONET_V7_RB_HP_TanhCritic": {  # Like VOLVONET_V7 with ResBlock based actors module
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_128|K1",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "VOLVONET_V7_RB_HP_Shallow_TanhCritic": {  # Like VOLVONET_V7 with ResBlock based actors module
        "ego_layers_structure": ["C1_64|K1",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_128|K1",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "RB1_128|K1|P0", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "AttentionNet_V1": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "T_1|2", "SA_M64|H4|N64", "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNet_V1_Slim": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_32|K1", "A_ReLU",
                                    "T_1|2", "SA_M32|H4|N32", "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNet_V1_Fat": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "T_1|2", "SA_M128|H4|N128", "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNet_V2": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M64|H4|N64",
                                    "SA_M64|H4|N64",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNet_V2_Slim": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_32|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M32|H4|N32",
                                    "SA_M32|H4|N32",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNet_V2_Fat": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M128|H4|N128",
                                    "SA_M128|H4|N128",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNet_V3": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M64|H4|N128",
                                    "SA_M64|H4|N128",
                                    "SA_M64|H4|N128",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "POINTNET_V1": {  # Similar to VolvoNet but concatenating the global vector as an intermediate operator
        "ego_layers_structure": ["C1_64|K1", "A_ReLU",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU", "RB1_64|K1", "A_ReLU", "ConcatWithMaxedLastDim",
                                    "C1_128|K1", "A_ReLU", "RB1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "POINTNET_V2": {  # Like POINTNET_V1 with additional interaction layer
        "ego_layers_structure": ["C1_64|K1", "A_ReLU",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU", "RB1_64|K1", "A_ReLU", "ConcatWithMaxedLastDim",
                                    "C1_128|K1", "A_ReLU", "RB1_128|K1", "A_ReLU", "ConcatWithMaxedLastDim",
                                    "C1_256|K1", "A_ReLU", "RB1_256|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "POINTNET_V3": {  # Like POINTNET_V2 with contant sized bottleneck
        "ego_layers_structure": ["C1_64|K1", "A_ReLU",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU", "RB1_64|K1", "A_ReLU", "ConcatWithMaxedLastDim",
                                    "C1_64|K1", "A_ReLU", "RB1_64|K1", "A_ReLU", "ConcatWithMaxedLastDim",
                                    "C1_64|K1", "A_ReLU", "RB1_64|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "POINTNET_V3_Fat": {  # Like POINTNET_V3 with wider bottleneck
        "ego_layers_structure": ["C1_64|K1", "A_ReLU",
                                 "RB1_64|K1|P0", "A_ReLU",
                                 "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU", "RB1_128|K1", "A_ReLU", "ConcatWithMaxedLastDim",
                                    "C1_128|K1", "A_ReLU", "RB1_128|K1", "A_ReLU", "ConcatWithMaxedLastDim",
                                    "C1_128|K1", "A_ReLU", "RB1_128|K1", "A_ReLU",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },

    # WITH LAST ACTION ARCHITECTURES #

    "AttentionNetWithLastAction_V2": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M64|H4|N64",
                                    "SA_M64|H4|N64",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNetWithLastAction_V2b": {
        "ego_layers_structure": ["C1_128|K1", "RB1_128|K1|P0", "A_ReLU", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M64|H4|N64",
                                    "SA_M64|H4|N64",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNetWithLastAction_V2b_Tanh": {
        "ego_layers_structure": ["C1_128|K1", "RB1_128|K1|P0", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M64|H4|N64",
                                    "SA_M64|H4|N64",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "AttentionNetWithLastAction_V2c": {
        "ego_layers_structure": ["C1_64|K1", "RB1_64|K1|P0", "A_ReLU", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M128|H4|N128",
                                    "SA_M128|H4|N128",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNetWithLastAction_V3": {
        "ego_layers_structure": ["C1_128|K1", "RB1_128|K1|P0", "A_ReLU", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M128|H4|N128",
                                    "SA_M128|H4|N128",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNetWithLastAction_V3b": {
        "ego_layers_structure": ["C1_128|K1", "RB1_128|K1|P0", "A_ReLU", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M128|H4|N128",
                                    "SA_M128|H4|N128",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },
    "AttentionNetWithLastAction_V3c": {
        "ego_layers_structure": ["C1_128|K1", "RB1_128|K1|P0", "A_ReLU", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M128|H4|N128",
                                    "SA_M128|H4|N128",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_512", "A_ReLU", "FC_512", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Sigmoid"]
    },

    # ARCHITECTURES FOR PAPER #
    # FZI architecture (Use with DualInputsActorCriticModel custom model)
    "FZI_Paper_V1": {
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["F"],
        "shared_layers_structure": ["FC_512", "A_ReLU", "FC_512", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_64", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_64", "A_ReLU", "FC_1", "A_Tanh"]
    },

    # Volvo architecture - same as paper except Tanh activation for critic
    "VOLVO_Paper_V1": {  # As in https://arxiv.org/abs/1905.02680 (Use with DualInputsActorCriticModel custom model)
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_32|K1", "A_ReLU", "C1_32|K1", "A_ReLU", "MaxLastDim", "F"],
        "shared_layers_structure": ["FC_64", "A_ReLU", "FC_64", "A_ReLU"],
        "actor_head_structure": ["FC_32", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_32", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "VOLVO_Paper_V2": {  # 4x neurons than VOLVO_Paper_V1
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU", "C1_128|K1", "A_ReLU", "MaxLastDim", "F"],
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "VOLVO_Paper_V3": {  # 8x neurons than VOLVO_Paper_V1
        "ego_layers_structure": ["F"],
        "actors_layers_structure": ["C1_256|K1", "A_ReLU", "C1_256|K1", "A_ReLU", "MaxLastDim", "F"],
        "shared_layers_structure": ["FC_512", "A_ReLU", "FC_512", "A_ReLU"],
        "actor_head_structure": ["FC_256", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_256", "A_ReLU", "FC_1", "A_Tanh"]
    },

    # Our paper architecture - use with AttentionActorCriticModelV2 and 4lane_lcftr_full_kinematics_with_goal encoder
    "Attention_Paper_Tanh": {
        "ego_layers_structure": ["C1_128|K1", "RB1_128|K1|P0", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M64|H4|N64",
                                    "SA_M64|H4|N64",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "Attention_Paper_Tanh_LeanerEgo": {
        "ego_layers_structure": ["C1_64|K1", "A_ReLU", "RB1_64|K1|P0", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_64|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M64|H4|N64",
                                    "SA_M64|H4|N64",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "Attention_Paper_Tanh_LeanerEgoOneAttention": {
        "ego_layers_structure": ["C1_64|K1", "A_ReLU", "RB1_64|K1|P0", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M128|H4|N128",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "A_ReLU", "FC_256", "A_ReLU"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    },
    "Attention_Paper_SingleAttentionSingleSkipFC": {
        "ego_layers_structure": ["C1_64|K1", "A_ReLU", "RB1_64|K1|P0", "F"],
        "last_action_layers_structure": ["EMB_32|E{out}", "A_ReLU", "FC_32", "A_ReLU", "F"],
        "actors_layers_structure": ["C1_128|K1", "A_ReLU",
                                    "T_1|2",
                                    "SA_M128|H4|N128",
                                    "T_1|2",
                                    "MaxLastDim", "F"],  # Compute global vector by MaxPooling over last dimension
        "shared_layers_structure": ["FC_256", "SingleSkipFC"],
        "actor_head_structure": ["FC_128", "A_ReLU", "FC_{out}"],
        "critic_head_structure": ["FC_128", "A_ReLU", "FC_1", "A_Tanh"]
    }

}
