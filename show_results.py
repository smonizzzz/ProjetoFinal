import json

h = json.load(open('results/scoliosis_hrnet/history.json'))
t = json.load(open('results/scoliosis_hrnet/test_results.json'))

val = h['val']
best_i = min(range(len(val)), key=lambda i: val[i]['cmae'])
b = val[best_i]

print(f'\n  Epoch melhor: {best_i+1}')
print(f'  {"Angulo":<14} {"Validacao":>10} {"Teste":>10}')
print(f'  {"-"*36}')
print(f'  {"CMAE medio":<14} {b["cmae"]:>9.2f}°  {t["cmae"]:>9.2f}°')
print(f'  {"Toracico T1":<14} {b["cmae_thoracic1"]:>9.2f}°  {t["cmae_thoracic1"]:>9.2f}°')
print(f'  {"Toracico T2":<14} {b["cmae_thoracic2"]:>9.2f}°  {t["cmae_thoracic2"]:>9.2f}°')
print(f'  {"Lombar":<14} {b["cmae_lumbar"]:>9.2f}°  {t["cmae_lumbar"]:>9.2f}°')
