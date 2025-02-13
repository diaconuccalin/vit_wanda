from timm.models import create_model


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()

    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, pref=""):
        local_metadata = {} if metadata is None else metadata.get(pref[:-1], {})

        module._load_from_state_dict(
            state_dict,
            pref,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        for name, child in module._modules.items():
            if child is not None:
                load(child, pref + name + ".")

    load(model, pref=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []

    for key in missing_keys:
        keep_flag = True

        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break

        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )

    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )

    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )

    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


def build_model(args, pretrained=False):
    return create_model(
        args.model,
        pretrained=pretrained,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        drop_rate=args.dropout,
    )
