import logging
from pathlib import Path
import subprocess
import os


monster_name = {
    "弧光锋卫长" : "Arc_Frontliner_Leader",
    "炮击组长" : "Mortar_Gunner_Leader",
    "复仇者" : "Hateful_Avenger",
    "重装防御者" : "Heavy_Defender",
    "“庞贝”" : "Pompeii",
    "冰原术师" : "Icefield_Caster",
    "沸血骑士团精锐" : "Bloodboil_Knightclub_Elite",
    "高塔术师" : "Spire_Caster",
    "固海凿石者" : "Ocean_Stonecutter",
    # "呼啸骑士团学徒" : "Roar_Knightclub_Trainee",
    "湖畔志愿者" : "Lakeside_Volunteer",
    "杰斯顿·威廉姆斯" : "Jesselton_Williams",
    "酸液源石虫·α" : "Acid_Originium_Slug_α",
    "神射手囚犯" : "Elite_Sniper_Prisoner",
    "拳手囚犯" : "Pugilist_Prisoner",
    "染污躯壳" : "Tainted_Carcass",
    "泥岩巨像" : "Mudrock_Colossus",
    "狂暴的猎狗pro" : "Rabid_Hound_Pro",
    "宿主拾荒者" : "Possessed_Veteran_Junkman",
    "狂暴宿主组长" : "Enraged_Possessed_Leader",
    "萨卡兹大剑手" : "Sarkaz_Greatswordsman",
    "矿脉守卫" : "Vein_Guardian",
    "山海众窥魅人" : "Shanhaizhong_Seer",
    "提亚卡乌好战者" : "Tiacauh_Fanatic",
    "码头水手" : "Dockworker",
    "变异巨岩蛛" : "Mutant_Giant_Rock_Spider",
    "萨卡兹子裔链术师" : "Sarkaz_Heirbearer_Chain_Caster",
    "温顺的武装驮兽" : "Armored_Burdenbeast",
    "木裂战士" : "Shattered_Champion",
    "深溟裂礁者" : "Nethersea_Reefbreaker",
    "山雪鬼" : "Tschäggättä",
    "高普尼克" : "Gopnik",
    "冰爆源石虫" : "Infused_Glacial_Originium_Slug",
    "反装甲步兵" : "Anti-Armor_Infantry",
    "“钳钳生风”" : "Consortium_of_Pincers",
    "富营养的穿刺者" : "Nourished_Piercer",
    "高级武装人员" : "Senior_Armed_Militant",
    "朗姆酒推荐者" : "Rum_Connoisseur",
    "烈酒级醒酒助手" : "Whiskey-Grade_Waker-Upper",
    "萨卡兹王庭军术师" : "Sarkaz_Royal_Court_Caster",
    # "灰尾香主" : "Greytail_Leader",
    "阵地击人手" : "Field_Bludgeoner",
    "源石畸变体" : "Originiutant",
    "提亚卡乌破坏王" : "Tiacauh_Annihilator",
    "逐腐兽" : "Rotchaser",
    "高能源石虫" : "Infused_Originium_Slug",
    "风情街“星术师”" : "Fashion_Street_Stellar_Caster",
    "田鼷力士" : "Fieldmus_Bruiser",
    "残党萨克斯手" : "Remnant_Saxophonist",
    "散华骑士团学徒" : "Nova_Knightclub_Trainee",
    "“阿咬”" : "Bitey",
    # "“门”" : "",
    "“投石机”" : "Catapult",
    # "“复仇者”" : '"Hateful Avenger"',
    # "扎罗，“狼之主”" : "Zaaro",
    # "“自在”" : "Free",
    # "灼热源石虫" : "Blazing_Originium_Slug",
    # "萨卡兹子裔责罚者" : "Sarkaz_Heirbearer_Punisher",
    # "" : "",
}

def extract_webm(webm_path: Path, output_folder: Path):
    extract_webm_cmd = f"ffmpeg -c:v libvpx -i \"{webm_path}\" \"{output_folder}/frame%d.png\""
    subprocess.run(extract_webm_cmd, shell=True, check=True)


def main():
    webm_folder = "./tools/battlefield_composite/monster_images"
    for webm_path in Path(webm_folder).glob("*.webm"):
        name_split = webm_path.parts[-1].split("-")
        print(name_split)
        if (name := monster_name.get(name_split[0])) is not None:
            output_directory = webm_path.parent / (name + "-" + name_split[3])
            print(output_directory)
            if not output_directory.exists():
                output_directory.mkdir(exist_ok=True)
                extract_webm(webm_path, output_directory)
            else:
                logging.warning(f"{output_directory} already exists")
        else:
            logging.error(f"Name: {name_split[0]} not found!")


if __name__ == "__main__":
    main()
