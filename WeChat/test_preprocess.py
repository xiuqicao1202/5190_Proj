# test_simple_preprocess.py
from preprocess import prepare_data

# Create test data
test_csv = """url
https://www.foxnews.com/lifestyle/jack-carrs-eisenhower-d-days-memo-noble-undertaking
https://www.foxnews.com/entertainment/bruce-willis-demi-moore-avoided-doing-one-thing-while-co-parenting-daughter-says
https://www.foxnews.com/politics/blinken-meets-with-qatars-prime-minister.print
https://www.foxnews.com/entertainment/emily-blunt-says-toes-curl-when-people-their-kids-want-act-want-say-dont-do-it
https://www.foxnews.com/media/the-view-co-host-cnn-commentator-ana-navarro-host-night-2-democratic-national-convention
https://www.nbcnews.com/politics/donald-trump/trump-says-pardon-large-portion-jan-6-rioters-rcna83873
https://www.nbcnews.com/politics/congress/jan-6-committee-avoids-criticizing-law-enforcement-summary-rcna61935
https://www.nbcnews.com/select/shopping/affordable-portable-air-conditioners-ncna1266638
https://www.nbcnews.com/select/shopping/jbl-clip-5-review-rcna172453
https://www.nbcnews.com/select/shopping/best-disposable-face-mask-ncna1252865
https://www.nbcnews.com/news/us-news/asheville-north-carolina-helene-damage-rcna173131
https://www.nbcnews.com/news/world/senior-hamas-leader-killed-beirut-was-key-figure-patched-ties-iran-rcna131993
https://www.nbcnews.com/politics/2024-election/abortion-rights-ballot-measure-south-dakota-rcna152746
https://www.nbcnews.com/politics/2024-election/totally-illegal-trump-escalates-rhetoric-outlawing-political-dissent-c-rcna174280
https://www.nbcnews.com/politics/justice-department/qanon-believer-chased-officer-capitol-sentenced-rcna61946
https://www.nbcnews.com/politics/elections/desantis-signs-bill-creating-election-police-unit-florida-rcna25941
https://www.nbcnews.com/investigations/russia-aims-undermine-biden-november-election-intel-officials-say-rcna161011
"""

# Save to a temporary file (or use StringIO)
with open('test.csv', 'w') as f:
    f.write(test_csv)

# Test
X, y = prepare_data('test.csv')

print("Results:")
print(f"Number of samples: {len(X)}")
print("\nProcessed data:")
for i, (title, label) in enumerate(zip(X, y)):
    print(f"{i+1}. Title: '{title}'")
    print(f"   Label: {label}")
    print()