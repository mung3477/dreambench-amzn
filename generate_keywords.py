import json

themes = {
    "Fashion": {
        "text-label": ['graphic tee', 'slogan', 'lettering', 'typography', 'printed shirt', 'text print', 'label', 'tag'],
        "logo": ['monogram', 'patch', 'badge', 'embroidered', 'trademark', 'logo', 'brand', 'emblem', 'sneaker', 'shoe', 'shirt', 'backpack', 'cap'],
        "intricate-geometry": ['lace', 'mesh', 'buckle', 'zipper', 'chain', 'pendant', 'jewelry', 'chronograph', 'woven', 'knit', 'watch']
    },
    "BeautyHealth": {
        "text-label": ['bottle', 'packaging', 'box', 'tube', 'can', 'vitamin', 'supplement', 'label', 'text', 'print', 'ingredients', 'instructions'],
        "logo": ['brand logo', 'signature', 'emblem', 'trademark', 'makeup palette', 'cosmetic brand'],
        "intricate-geometry": ['trimmer', 'clipper', 'shaver', 'brush bristles', 'pump mechanism', 'spray nozzle', 'compact case']
    },
    "HomeGarden": {
        "text-label": ['control panel', 'instruction label', 'warning label', 'brand name', 'digital display', 'packaging'],
        "logo": ['appliance brand', 'maker mark', 'trademark', 'logo', 'emblem'],
        "intricate-geometry": ['machine', 'engine', 'tool', 'gears', 'blades', 'motor', 'pump', 'grill', 'ventilation', 'circuit']
    },
    "TechGames": {
        "text-label": ['keyboard', 'keypad', 'screen text', 'digital interface', 'packaging', 'box art', 'game cover'],
        "logo": ['tech brand', 'manufacturer logo', 'trademark', 'console logo', 'software logo', 'emblem'],
        "intricate-geometry": ['motherboard', 'circuit', 'chip', 'controller', 'joystick', 'ports', 'wiring', 'lens', 'camera module', 'cooling fan']
    },
    "Media": {
        "text-label": ['book cover', 'title', 'author', 'typography', 'magazine cover', 'album art', 'tracklist', 'spine'],
        "logo": ['publisher logo', 'record label', 'movie studio', 'trademark'],
        "intricate-geometry": ['vinyl grooves', 'cd surface', 'intricate cover art', 'disc packaging', 'film reel']
    },
    "HobbiesToys": {
        "text-label": ['game board', 'cards', 'instructions', 'packaging', 'box', 'coin engraving', 'sheet music'],
        "logo": ['toy brand', 'maker mark', 'trademark', 'emblem'],
        "intricate-geometry": ['action figure', 'building blocks', 'model kit', 'puzzle', 'musical instrument', 'strings', 'keys', 'coin relief', 'sewing machine', 'craft tool']
    },
    "AutoIndustrial": {
        "text-label": ['warning label', 'specifications', 'part number', 'gauge reading', 'packaging', 'instruction panel'],
        "logo": ['car make', 'brand emblem', 'manufacturer logo', 'trademark'],
        "intricate-geometry": ['engine part', 'gears', 'suspension', 'circuit board', 'microscope', 'caliper', 'machinery', 'tread', 'grille']
    },
    "FoodPets": {
        "text-label": ['nutrition facts', 'ingredients', 'food packaging', 'bag', 'can', 'box', 'wrapper', 'pet food label'],
        "logo": ['food brand', 'trademark', 'logo', 'emblem'],
        "intricate-geometry": ['kibble shape', 'pet toy structure', 'aquarium filter', 'intricate food mold', 'cage mesh']
    },
    "Special": {
        "text-label": ['gift card text', 'personalized', 'engraved', 'baby monitor text', 'sign', 'calligraphy', 'custom text'],
        "logo": ['brand tag', 'artisan stamp', 'maker mark', 'trademark', 'logo'],
        "intricate-geometry": ['stroller mechanism', 'car seat buckle', 'filigree', 'macrame', 'bicycle gear', 'fishing reel', 'tent structure']
    }
}

category_to_theme = {
    "raw_meta_Amazon_Fashion": "Fashion",
    "raw_meta_Clothing_Shoes_and_Jewelry": "Fashion",
    "raw_meta_All_Beauty": "BeautyHealth",
    "raw_meta_Beauty_and_Personal_Care": "BeautyHealth",
    "raw_meta_Health_and_Household": "BeautyHealth",
    "raw_meta_Health_and_Personal_Care": "BeautyHealth",
    "raw_meta_Appliances": "HomeGarden",
    "raw_meta_Home_and_Kitchen": "HomeGarden",
    "raw_meta_Patio_Lawn_and_Garden": "HomeGarden",
    "raw_meta_Tools_and_Home_Improvement": "HomeGarden",
    "raw_meta_Cell_Phones_and_Accessories": "TechGames",
    "raw_meta_Electronics": "TechGames",
    "raw_meta_Software": "TechGames",
    "raw_meta_Video_Games": "TechGames",
    "raw_meta_Books": "Media",
    "raw_meta_CDs_and_Vinyl": "Media",
    "raw_meta_Digital_Music": "Media",
    "raw_meta_Kindle_Store": "Media",
    "raw_meta_Magazine_Subscriptions": "Media",
    "raw_meta_Movies_and_TV": "Media",
    "raw_meta_Arts_Crafts_and_Sewing": "HobbiesToys",
    "raw_meta_Collectible_Coins": "HobbiesToys",
    "raw_meta_Musical_Instruments": "HobbiesToys",
    "raw_meta_Toys_and_Games": "HobbiesToys",
    "raw_meta_Automotive": "AutoIndustrial",
    "raw_meta_Industrial_and_Scientific": "AutoIndustrial",
    "raw_meta_Grocery_and_Gourmet_Food": "FoodPets",
    "raw_meta_Pet_Supplies": "FoodPets",
    "raw_meta_Baby_Products": "Special",
    "raw_meta_Gift_Cards": "Special",
    "raw_meta_Handmade_Products": "Special",
    "raw_meta_Sports_and_Outdoors": "Special"
}

def main():
    final_mapping = {}
    for cat, theme in category_to_theme.items():
        final_mapping[cat] = themes[theme]
        
    with open("category_keywords.json", "w") as f:
        json.dump(final_mapping, f, indent=4)
        
    print(f"Generated keywords for {len(final_mapping)} categories.")

if __name__ == "__main__":
    main()
