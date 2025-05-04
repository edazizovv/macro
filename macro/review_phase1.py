#
import os


#
import pandas

#

#
def recompute(
    target,
    phase,
):

    d = "../data/data_folds/"
    g_train = f"data_train_ph{phase}"
    g_test = f"data_test_ph{phase}"

    data = {
        "train": {str(i): [] for i in range(10)},
        "test": {str(i): [] for i in range(10)},
    }
    for z in os.listdir(d):
        if g_train in z:
            fold_n = z[z.index(".xlsx") - 1:z.index(".xlsx")]
            sliced = pandas.read_excel(f"{d}{z}")
            sliced = sliced.rename(columns={sliced.columns[0]: "DATE"})
            sliced["DATE"] = pandas.to_datetime(sliced["DATE"])
            sliced = sliced.set_index("DATE")
            data["train"][fold_n].append(sliced)
        if g_test in z:
            fold_n = z[z.index(".xlsx") - 1:z.index(".xlsx")]
            sliced = pandas.read_excel(f"{d}{z}")
            sliced = sliced.rename(columns={sliced.columns[0]: "DATE"})
            sliced["DATE"] = pandas.to_datetime(sliced["DATE"])
            sliced = sliced.set_index("DATE")
            data["test"][fold_n].append(sliced)
    data_joint = []
    for t in data.keys():
        for k in data[t].keys():
            data[t][k] = pandas.concat(data[t][k], axis=1, ignore_index=False)
            data[t][k]["role"] = t
            data[t][k]["fold"] = k
            data_joint.append(data[t][k])
    data_joint = pandas.concat(data_joint, axis=0, ignore_index=False)
    data_joint[target] = data_joint[target].shift(-1)

    return data_joint


# target = "IVV_aggmean_pct"
# data_joint = recompute(target, 1)
