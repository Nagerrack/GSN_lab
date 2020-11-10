def copyModel2Model(model_source, model_target, certain_layer=""):
    for l_tg, l_sr in zip(model_target.layers, model_source.layers):
        wk0 = l_sr.get_weights()
        l_tg.set_weights(wk0)
        if l_tg.name == certain_layer:
            break
    print("model source was copied into model target")
