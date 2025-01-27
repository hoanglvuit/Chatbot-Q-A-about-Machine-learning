from semantic_router import Route
from semantic_router.encoders import CohereEncoder
from semantic_router.layer import RouteLayer
from preprocessing import * 
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Định nghĩa các routes (giữ nguyên như trước)
vu_huu_tiep = Route(
        name="0",
        utterances=keywords[0],
    )
loi_tac_gia =  Route(
        name="1",
        utterances=keywords[1],
    )
on_tap_dai_so_tuyen_tinh = Route(
        name="2",
        utterances=keywords[2],
    )
giai_tich_ma_tran = Route(
        name="3",
        utterances=keywords[3],
    )
on_tap_xac_suat = Route(
        name="4",
        utterances=keywords[4],
    )
maximum_likelihood_va_maximum_a_posteriori  = Route(
        name="5",
        utterances=keywords[5],
    )
cac_khai_niem_co_ban = Route(
        name="6",
        utterances=keywords[6],
    )
gioi_thieu_ve_feature_engineering = Route(
        name="7",
        utterances=keywords[7] ,
    )
linear_regression = Route(
        name="8",
        utterances=keywords[8],
    )
overfitting = Route(
        name="9",
        utterances=keywords[9],
    )
k_nearest_neighbors = Route(
        name="10",
        utterances=keywords[10],
    )
k_means_clustering = Route(
        name="11",
        utterances=keywords[11],
    )
naive_bayes_classifier = Route(
    name = '12' ,
    utterances = keywords[12] ,
)
gradient_descent = Route(
        name="13",
        utterances=keywords[13],
    )
perceptron_learning_algorithm = Route(
        name="14",
        utterances=keywords[14],
    )
logistic_regression = Route(
        name="15",
        utterances=keywords[15],
    )
softmax_regression = Route(
        name="16",
        utterances=keywords[16],
    )
multilayer_neural_network_va_backpropagation = Route(
        name="17",
        utterances=keywords[17],
    )
content_based_recommendation_system = Route(
        name="18",
        utterances=keywords[18],
    )
neighborhood_based_collaborative_filtering = Route(
        name="19",
        utterances=keywords[19],
    )
matrix_factorization_collaborative_filtering = Route(
        name="20",
        utterances=keywords[20],
    )
singular_value_decomposition = Route(
        name="21",
        utterances=keywords[21],
    )
principal_component_analysis = Route(
        name="22",
        utterances=keywords[22],
    )
linear_discriminant_analysis = Route(
        name="23",
        utterances=keywords[23],
    )
tap_loi_va_ham_loi = Route(
        name="24",
        utterances=keywords[24],
    )
bai_toan_toi_uu_loi = Route(
        name="25",
        utterances=keywords[25],
    )
duality = Route(
        name="26",
        utterances=keywords[26],
    )
support_vector_machine = Route(
        name="27",
        utterances=keywords[27],
    )
soft_margin_support_vector_machine = Route(
        name="28",
        utterances=keywords[28],
    )
kernel_support_vector_machine = Route(
        name="29",
        utterances=keywords[29],
    )
multi_class_support_vector_machine = Route(
        name="30",
        utterances=keywords[30],
    )


routes = [vu_huu_tiep,loi_tac_gia,on_tap_dai_so_tuyen_tinh,naive_bayes_classifier,giai_tich_ma_tran,on_tap_xac_suat,maximum_likelihood_va_maximum_a_posteriori,cac_khai_niem_co_ban,gioi_thieu_ve_feature_engineering,linear_regression,overfitting,k_nearest_neighbors,k_means_clustering,gradient_descent,perceptron_learning_algorithm,logistic_regression,softmax_regression,multilayer_neural_network_va_backpropagation,content_based_recommendation_system,neighborhood_based_collaborative_filtering,matrix_factorization_collaborative_filtering,singular_value_decomposition,principal_component_analysis,linear_discriminant_analysis,tap_loi_va_ham_loi,bai_toan_toi_uu_loi,duality,support_vector_machine,soft_margin_support_vector_machine,kernel_support_vector_machine,multi_class_support_vector_machine]

# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
encoder = CohereEncoder()

# Tạo RouteLayer
route_layer = RouteLayer(encoder=encoder, routes=routes,aggregation = 'max',)

