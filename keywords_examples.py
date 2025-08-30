spam_keywords = [
    "visit", "discounts", "promo", "deal", "coupon", "free", "limited time",
    "offer", "special", "sale", "buy now", "click here", "sign up", "subscribe",
    "follow us", "best service", "affordable prices", "quality products",
    "leading provider", "check out", ".com", ".net", "www.", "link in bio",
    "call now", "contact us", "exclusive", "guaranteed", "low prices", "new stock"
]

spam_reviews = [
    "Best sushi in town! We have a buy-one-get-one-free deal this weekend only. Visit our website for details: www.sushihub.com.",
    "The coffee here is okay, but if you want to experience real coffee, come to our new cafe downtown! We have a special introductory offer for new customers.",
    "I had a great time at this restaurant, and you can, too! Follow me on Instagram for a promo code to save 10% on your next meal.",
    "Looking for an affordable car wash? Check out our special deals! Mention this review to get a discount.",
    "This place is so beautiful, but for true beauty tips and tricks, subscribe to my YouTube channel. Link in bio!",
    "The food here is good, but for the best dining experience and an exclusive discount, visit www.dinerdeals.com.",
    "I don't usually write reviews, but their service was top-notch! Get a free consultation by calling their number today.",
    "The staff was so friendly! For even friendlier service and amazing deals on electronics, visit us at TechWorld.com.",
    "This is a nice park, but if you want to find the best hiking gear, visit our store for a limited-time sale!",
    "The hotel room was clean. For the best hotel cleaning services in the city, contact us for a free quote!",
    "This gym is fine, but for a true fitness journey, sign up for our online coaching program. We're offering a 50% discount.",
    "I had a wonderful experience! To get more wonderful experiences and great savings, download my app, 'Savings Plus'!",
    "The pizza was good, but for the best pizza deals, visit our page on Facebook. We're running a contest!",
    "This is a great dentist! For a whiter smile, check out my affiliate link for a whitening kit: www.whitenow.net/promo.",
    "I love this store! For even more amazing fashion, follow my TikTok account and get a discount code in my bio.",
    "The concert hall was loud. For the highest quality audio equipment, visit us at LoudSound.com.",
    "The burgers are a bit greasy. For healthy, delicious meal prep, visit my blog and get a free recipe book.",
    "This hotel is beautiful, but our resort offers better views and a buy-one-get-one-free spa treatment. Call us today!",
    "The service was slow. For quick and reliable services, visit QuickFix.com. We're having a 20% off sale this week.",
    "I had a decent haircut. For a truly amazing look, check out our salon. We have a special offer for new clients!"
]

irrelevant_keywords = []

irrelevant_reviews = [
    "The coffee here is okay, but I can't believe how expensive my new laptop was. I wish I had waited for the Black Friday sale. Anyway, the chairs here are a bit wobbly.",
    "I came here with my friend, and we had a great time catching up. She was telling me all about her new puppy, which is so cute! The food was fine, I guess.",
    "This is a beautiful hotel, but I really hate the current political climate. It just makes me so mad. The hotel room was clean, though.",
    "The food was delicious, but my favorite sports team lost last night, and I'm so bummed. The waiter was nice.",
    "I had a great time at the bar. By the way, my cousin just got engaged, and I'm so happy for her!",
    "The park is nice for a walk. I'm going to get my nails done later, I can't wait.",
    "This store has a good selection. I just finished my favorite TV show, and I'm so sad it's over.",
    "I went to this restaurant for a birthday party. I had a lot of fun, but my job is so stressful right now.",
    "The front desk was friendly. I was thinking about the new video game that's coming out, it looks so cool.",
    "The zoo was great for the kids. My son just started a new school year, and I'm so proud of him.",
    "The line was a bit long. On a different note, my dog learned a new trick yesterday!",
    "I had a decent haircut here. The weather is supposed to be great this weekend for a hike.",
    "The concert was loud. I saw the new movie, and it was amazing!",
    "This hotel has nice rooms. The economy is a mess, though.",
    "The library is quiet. I'm thinking about what to make for dinner tonight.",
    "The food was good, but I'm so tired from work this week.",
    "The park is nice. I can't believe the price of gas right now.",
    "The gym is a good place to work out. I'm trying to decide what to watch on Netflix later.",
    "The atmosphere was cozy. My cat has been acting weird lately.",
    "I had a great time here. I'm going on vacation next week, I'm so excited!"
]

non_visitor_keywords = [
    "never been here", "I haven't visited", "I've heard", "a friend told me",
    "read online that", "I saw a report", "I wouldn't go", "if I were to go",
    "seems like a bad place", "looks awful from the outside", "I can tell",
    "based on what I saw", "after seeing pictures", "heard from a friend",
    "I saw a news story", "my cousin told me", "from what I hear",
    "I'm not going to visit", "I've decided not to go",
    "from the looks of it", "I read about", "I heard a rumor",
    "I just drove by"
]

non_visitor_reviews = [
    "Never been here, but I saw a news report about a health code violation. I would never eat at a place like this. It looks disgusting.",
    "My cousin told me this place is terrible. She said the staff was rude and the food was cold. I'm going to give it one star just based on that.",
    "I was thinking of going here, but after seeing the long lines in a picture online, I decided against it. I'm giving it a one-star review because that's just poor management.",
    "I haven't visited yet, but I heard from a friend that the service is terrible. It seems like a bad place to go.",
    "I can't imagine anyone having a good time here. From the looks of it, this place is run-down and dirty. I'm giving it one star.",
    "I read an article about how this place mistreats its employees. I wouldn't support a business like that. One star.",
    "I drove by once and the building looks so old and unappealing. I'm not going to visit, and I'm warning others not to either.",
    "I've heard the prices are way too high for what you get. I'm not even going to bother trying it. One star.",
    "This place looks like a tourist trap. I would avoid it at all costs. I haven't been, but I can tell.",
    "The reviews I've read all say the same thing: bad service. I'm just adding to the chorus and giving a one-star review.",
    "My uncle told me the food is bland and overpriced. Based on that, I'm giving it one star.",
    "I heard a rumor that this place is closing down soon. I'm giving it one star because that seems unprofessional.",
    "I saw a picture of a messy dining area. If I were to go, I'm sure it would be dirty. One star.",
    "This place looks like a total scam. I'm giving it one star just to warn people.",
    "I've decided not to go here after reading about their no-return policy. That's just a terrible business practice. One star.",
    "I can't believe this place has so many positive reviews. I'm guessing they're fake. One star from me.",
    "I've never been, but the building looks abandoned and creepy. I wouldn't feel safe going here.",
    "I heard from a friend that the wait time is over two hours on the weekends. I'm giving it one star for that reason.",
    "Based on what I've seen on social media, this place is nothing but a disappointment.",
    "The pictures online look very unprofessional. I'm just giving it one star because it doesn't look like a reputable business."
]