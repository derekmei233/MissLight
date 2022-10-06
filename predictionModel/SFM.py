from utils.preparation import get_road_adj, inter2edge_slice, mask_op, get_mask_matrix, reconstruct_data_slice


# sfm_prediction only needs one time slice, redesign mask_op and construct_data
def mask_op_slice():
    pass

class SFM_predictor(object):
    def __init__(self):
        super(SFM_predictor, self).__init__()

    def predict(self, states, phases, relation, mask_pos, mask_matrix, adj_matrix, mode='select'):
        masked = inter2edge_slice(relation, states, phases, mask_pos)
        infer = mask_op(masked, mask_matrix, adj_matrix, mode)
        #edge_feature = infer * masked * -1 for debug use only
        edge_feature = infer + masked
        prediction = reconstruct_data_slice(edge_feature, phases, relation)
        return prediction

    def make_model(self, **kwargs):
        return self

    """
    def sfm_loss():
        pkl_file = open(file, 'rb')
        data = pickle.load(pkl_file)
        # (360,N,11)
        road_feature = data['road_feature']
        road_update = data['road_update']
        adj_road = get_road_adj(relation)

        select_data = np.expand_dims(road_feature[0],0).transpose(0,2,3,1)
        select_data = mask_op(select_data, road_update, adj_road,'select')
        y_predict = select_data[:,:,:3,:len(road_feature[0])-1]
        y_true = np.expand_dims(road_feature[0],0).transpose(0,2,3,1)[:,:,:3,1:]
        y_predict_re = reconstruct_data(y_predict, relation_file, 'cuda:0')
        y_true_re = reconstruct_data(y_true, relation_file, 'cuda:0')
        mae_result = mae(y_true_re,y_predict_re)
        rmse_result = rmse(y_true_re,y_predict_re)
        mape_result = mape(y_true_re,y_predict_re)
        print("Missing Intersection:")
        print("mae:",mae_result)
        print("rmse:",rmse_result)
        print("mape:",mape_result)
        print("done")
        """
