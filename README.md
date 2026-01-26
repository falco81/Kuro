I quickly made scripts for bulk editing of shops in .kurodlc.json

Example:
the first utility lists all unique ShopItem IDs in the .kurodlc.json file

python find_unique_shop_item_id.py FalcoDLC.kurodlc.json

[4550, 4551, 4552]

I will then manually save these to the configuration file for example "create_shops_template.json", including the IDs of all new shops where I want these IDs.

{
    "item_ids": [4550, 4551, 4552],
    "shop_ids": [21, 22, 23, 248]
}

The second script then generates a new definition of "ShopItem" which I can then replace with the original one in the .kurodlc.json file.

python create_shops.py create_shops_template.json
Success: File 'output_create_shops_template.json' was created successfully.

{
    "ShopItem": [
        {
            "shop_id": 21,
            "item_id": 4550,
            "unknown": 1,
            "start_scena_flags": [],
            "empty1": 0,
            "end_scena_flags": [],
            "int2": 0
        },
        {
.
.
.
.
