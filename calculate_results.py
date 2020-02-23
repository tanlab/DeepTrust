import pandas as pd
import os

deep_trust_acc = 0
gmm_acc = 0
kmeans_acc = 0
ward_acc = 0 

counter = 0

deep_trust_sil = 0
gmm_sil = 0
kmeans_sil = 0
ward_sil = 0 

for each_file in os.listdir():
     if each_file[:3] == "all":
          res = pd.read_pickle(each_file)
          
          deep_trust_acc += res['deepTrust']['acc']
          gmm_acc += res['gmm']['acc']
          kmeans_acc += res['kmeans']['acc']
          ward_acc += res['ward']['acc']

          deep_trust_sil += res['deepTrust']['sil']
          gmm_sil += res['gmm']['sil']
          kmeans_sil += res['kmeans']['sil']
          ward_sil += res['ward']['sil']
          counter += 1


print("deep acc: ",  deep_trust_acc / counter)
print("gmm acc: ", gmm_acc / counter)
print("kmeans acc: ",  kmeans_acc / counter)
print("ward acc: ", ward_acc / counter)

print("deep sil: ", deep_trust_sil / counter )
print("gmm sil: ",  gmm_sil / counter )
print("kmeans sil: ",  kmeans_sil / counter )
print("ward sil :", ward_sil/counter)
