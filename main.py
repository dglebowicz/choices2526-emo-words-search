from emo_words_search import EmoWordsFinder
import os
import pandas as pd

def main():
    portfolios_dir = 'Portfolios'
    aggregate_results = pd.DataFrame(columns = ['Portfolio_id','Emotion','Word','Sentence','Sentence_id','Search_method','Validation_method'])
    for i,file in enumerate(sorted(os.listdir(portfolios_dir))):
        
        emofinder = EmoWordsFinder(generate_test=False,loadpath=os.path.join(portfolios_dir, file))
        emofinder.find_emotional_words_by_stem(method='manual')
        emofinder.search_results_df['Portfolio_id'] = i+1        
        aggregate_results = pd.concat([aggregate_results, emofinder.search_results_df], ignore_index=True)
    aggregate_results.to_csv('emo_search_aggregate_results.csv', index=False)

if __name__ == "__main__":
    main()
