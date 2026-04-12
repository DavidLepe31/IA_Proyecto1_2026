from datasets import load_dataset

ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')
ds['train'].to_csv('bitext_dataset.csv', index=False)

print('Listo: bitext_dataset.csv')