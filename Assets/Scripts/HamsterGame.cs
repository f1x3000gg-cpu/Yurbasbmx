using System;
using System.Collections.Generic;
using UnityEngine;

namespace HamsterFlip
{
    public sealed class HamsterTaleBootstrap : MonoBehaviour
    {
        const float W = 1920f;
        const float H = 1080f;

        enum Mode { Title, Story, Explore, Dialogue, Battle, Ending }
        enum BattlePhase { Message, Menu, EnemyTurn, Defeat }

        sealed class Bullet
        {
            public Vector2 position;
            public Vector2 velocity;
            public float radius;
        }

        Mode mode;
        BattlePhase battlePhase;

        Texture2D pixel;
        Texture2D hamsterSprite;
        Texture2D moleSprite;
        Texture2D guardianSprite;
        Texture2D seedSprite;

        GUIStyle titleStyle;
        GUIStyle headingStyle;
        GUIStyle bodyStyle;
        GUIStyle smallStyle;
        GUIStyle buttonStyle;
        GUIStyle centerStyle;

        readonly Color ink = new Color(0.055f, 0.047f, 0.075f);
        readonly Color paper = new Color(0.94f, 0.86f, 0.66f);
        readonly Color gold = new Color(1f, 0.73f, 0.22f);
        readonly Color rose = new Color(0.87f, 0.25f, 0.31f);
        readonly Color moss = new Color(0.19f, 0.34f, 0.25f);
        readonly Color blue = new Color(0.22f, 0.42f, 0.62f);

        readonly Rect startButton = new Rect(710, 790, 500, 130);
        readonly Rect nextButton = new Rect(1510, 875, 280, 105);
        readonly Rect actionButton = new Rect(1540, 760, 270, 210);
        readonly Rect upButton = new Rect(205, 760, 130, 130);
        readonly Rect downButton = new Rect(205, 900, 130, 130);
        readonly Rect leftButton = new Rect(65, 900, 130, 130);
        readonly Rect rightButton = new Rect(345, 900, 130, 130);
        readonly Rect battleBox = new Rect(545, 245, 830, 440);
        readonly Rect battleTalk = new Rect(90, 860, 400, 125);
        readonly Rect battleSpare = new Rect(535, 860, 400, 125);
        readonly Rect battleSnack = new Rect(980, 860, 400, 125);
        readonly Rect battleAttack = new Rect(1425, 860, 400, 125);
        readonly Rect restartButton = new Rect(710, 820, 500, 125);

        readonly string[] introLines =
        {
            "Ночью в Норной Долине погас Последний Фонарь.",
            "Из кладовой исчезло золотое семечко — запас тепла на всю зиму.",
            "Старшие закрыли двери. Никто не решился спуститься к старым банкам.",
            "Кроме маленького хомяка по имени Пикс."
        };

        Vector2 playerPosition = new Vector2(420, 700);
        int storyIndex;
        int questStep;
        string[] dialogueLines;
        int dialogueIndex;
        Mode dialogueReturnMode;
        int dialogueAction;

        int playerHp;
        int enemyHp;
        int mercy;
        int snacks;
        string battleMessage;
        int battleNextAction;
        float enemyTurnRemaining;
        float spawnTimer;
        float invulnerability;
        int attackPattern;
        Vector2 soulPosition;
        readonly List<Bullet> bullets = new List<Bullet>();

        string[] endingLines;
        int endingIndex;
        string endingTitle;

        void Awake()
        {
            Application.targetFrameRate = 60;
            QualitySettings.vSyncCount = 0;
            Screen.orientation = ScreenOrientation.LandscapeLeft;
            CreateCamera();
            CreateTextures();
            ResetGame();
        }

        void CreateCamera()
        {
            GameObject cameraObject = new GameObject("Main Camera");
            cameraObject.tag = "MainCamera";
            Camera cameraComponent = cameraObject.AddComponent<Camera>();
            cameraComponent.clearFlags = CameraClearFlags.SolidColor;
            cameraComponent.backgroundColor = ink;
            cameraComponent.orthographic = true;
            cameraComponent.orthographicSize = 5f;
            cameraObject.AddComponent<AudioListener>();
        }

        void ResetGame()
        {
            mode = Mode.Title;
            storyIndex = 0;
            questStep = 0;
            playerPosition = new Vector2(420, 700);
            bullets.Clear();
        }

        void Update()
        {
            if (mode == Mode.Title)
            {
                if (TapIn(startButton))
                {
                    mode = Mode.Story;
                    storyIndex = 0;
                }
                return;
            }

            if (mode == Mode.Story)
            {
                if (TapIn(nextButton) || AnyTap())
                {
                    storyIndex++;
                    if (storyIndex >= introLines.Length)
                        mode = Mode.Explore;
                }
                return;
            }

            if (mode == Mode.Dialogue)
            {
                if (TapIn(nextButton) || AnyTap())
                {
                    dialogueIndex++;
                    if (dialogueIndex >= dialogueLines.Length)
                        FinishDialogue();
                }
                return;
            }

            if (mode == Mode.Explore)
            {
                UpdateExplore();
                return;
            }

            if (mode == Mode.Battle)
            {
                UpdateBattle();
                return;
            }

            if (mode == Mode.Ending)
            {
                if (endingIndex < endingLines.Length)
                {
                    if (TapIn(nextButton) || AnyTap())
                        endingIndex++;
                }
                else if (TapIn(restartButton))
                {
                    ResetGame();
                }
            }
        }

        void UpdateExplore()
        {
            Vector2 move = ReadMove();
            playerPosition += move * 330f * Time.deltaTime;
            playerPosition.x = Mathf.Clamp(playerPosition.x, 190f, 1715f);
            playerPosition.y = Mathf.Clamp(playerPosition.y, 190f, 780f);

            if (!TapIn(actionButton))
                return;

            float npcDistance = Vector2.Distance(playerPosition, new Vector2(950, 500));
            float bowlDistance = Vector2.Distance(playerPosition, new Vector2(480, 350));
            float doorDistance = Vector2.Distance(playerPosition, new Vector2(1570, 290));

            if (npcDistance < 210f)
            {
                if (questStep == 0)
                {
                    StartDialogue(new[]
                    {
                        "Тихон: Ты правда пойдёшь к старым банкам?",
                        "Тихон: Страж забрал семечко не из злости. Он чего-то боится.",
                        "Тихон: Не спеши драться. Иногда крышка открывается от правильных слов."
                    }, Mode.Explore, 1);
                }
                else
                {
                    StartDialogue(new[]
                    {
                        "Тихон: В его железном голосе всё ещё слышен страх.",
                        "Тихон: Помни — победить и уничтожить не одно и то же."
                    }, Mode.Explore, 0);
                }
            }
            else if (bowlDistance < 190f)
            {
                StartDialogue(new[]
                {
                    "Миска пуста.",
                    "На дне лежит одна скорлупка. Пикс кладёт её в карман на удачу."
                }, Mode.Explore, 0);
            }
            else if (doorDistance < 220f)
            {
                if (questStep == 0)
                {
                    StartDialogue(new[]
                    {
                        "За дверью что-то тяжело скребётся.",
                        "Пикс решает сначала спросить Тихона."
                    }, Mode.Explore, 0);
                }
                else
                {
                    StartDialogue(new[]
                    {
                        "Железная дверь открывается сама.",
                        "Из темноты выкатывается огромная банка с нарисованными глазами..."
                    }, Mode.Explore, 2);
                }
            }
        }

        void StartDialogue(string[] lines, Mode returnMode, int action)
        {
            dialogueLines = lines;
            dialogueIndex = 0;
            dialogueReturnMode = returnMode;
            dialogueAction = action;
            mode = Mode.Dialogue;
        }

        void FinishDialogue()
        {
            int action = dialogueAction;
            mode = dialogueReturnMode;
            dialogueAction = 0;

            if (action == 1)
                questStep = 1;
            else if (action == 2)
                StartBattle();
        }

        void StartBattle()
        {
            mode = Mode.Battle;
            battlePhase = BattlePhase.Message;
            playerHp = 20;
            enemyHp = 30;
            mercy = 0;
            snacks = 1;
            battleMessage = "СТРАЖ БАНКИ заслоняет золотое семечко.";
            battleNextAction = 0;
            soulPosition = battleBox.center;
            bullets.Clear();
        }

        void UpdateBattle()
        {
            if (battlePhase == BattlePhase.Message)
            {
                if (AnyTap())
                {
                    if (battleNextAction == 1)
                        BeginEnemyTurn();
                    else if (battleNextAction == 2)
                        StartPeacefulEnding();
                    else if (battleNextAction == 3)
                        StartDarkEnding();
                    else
                        battlePhase = BattlePhase.Menu;
                }
                return;
            }

            if (battlePhase == BattlePhase.Menu)
            {
                if (TapIn(battleTalk))
                {
                    mercy = Mathf.Min(100, mercy + 45);
                    battleMessage = mercy >= 100
                        ? "Пикс замечает: Страж дрожит не от злости, а от холода."
                        : "Пикс говорит, что пустая кладовая пугает всех. Страж слушает.";
                    battleNextAction = 1;
                    battlePhase = BattlePhase.Message;
                }
                else if (TapIn(battleSpare))
                {
                    if (mercy >= 100)
                    {
                        battleMessage = "Пикс опускает лапы. Страж медленно открывает крышку.";
                        battleNextAction = 2;
                    }
                    else
                    {
                        battleMessage = "Страж пока не верит, что его не уничтожат.";
                        battleNextAction = 1;
                    }
                    battlePhase = BattlePhase.Message;
                }
                else if (TapIn(battleSnack))
                {
                    if (snacks > 0)
                    {
                        snacks--;
                        playerHp = Mathf.Min(20, playerHp + 8);
                        battleMessage = "Пикс съедает запасную крошку. HP восстановлено.";
                    }
                    else
                    {
                        battleMessage = "В кармане осталась только скорлупка.";
                    }
                    battleNextAction = 1;
                    battlePhase = BattlePhase.Message;
                }
                else if (TapIn(battleAttack))
                {
                    int damage = UnityEngine.Random.Range(7, 11);
                    enemyHp = Mathf.Max(0, enemyHp - damage);
                    if (enemyHp <= 0)
                    {
                        battleMessage = "Последний удар раскалывает старую банку.";
                        battleNextAction = 3;
                    }
                    else
                    {
                        battleMessage = "Пикс ударяет по крышке. Страж теряет " + damage + " HP.";
                        battleNextAction = 1;
                    }
                    battlePhase = BattlePhase.Message;
                }
                return;
            }

            if (battlePhase == BattlePhase.EnemyTurn)
            {
                UpdateEnemyTurn();
                return;
            }

            if (battlePhase == BattlePhase.Defeat && TapIn(restartButton))
                StartBattle();
        }

        void BeginEnemyTurn()
        {
            battlePhase = BattlePhase.EnemyTurn;
            enemyTurnRemaining = 5.2f;
            spawnTimer = 0f;
            invulnerability = 0f;
            attackPattern = (attackPattern + 1) % 3;
            soulPosition = battleBox.center;
            bullets.Clear();
        }

        void UpdateEnemyTurn()
        {
            float dt = Time.deltaTime;
            enemyTurnRemaining -= dt;
            invulnerability = Mathf.Max(0f, invulnerability - dt);

            Vector2 directTouch;
            if (TryGetBattleDrag(out directTouch))
                soulPosition = directTouch;
            else
                soulPosition += ReadMove() * 470f * dt;

            soulPosition.x = Mathf.Clamp(soulPosition.x, battleBox.xMin + 24, battleBox.xMax - 24);
            soulPosition.y = Mathf.Clamp(soulPosition.y, battleBox.yMin + 24, battleBox.yMax - 24);

            spawnTimer -= dt;
            if (spawnTimer <= 0f)
            {
                SpawnBulletPattern();
                spawnTimer = attackPattern == 2 ? 0.22f : 0.34f;
            }

            for (int i = bullets.Count - 1; i >= 0; i--)
            {
                Bullet bullet = bullets[i];
                bullet.position += bullet.velocity * dt;

                if (Vector2.Distance(bullet.position, soulPosition) < bullet.radius + 18f && invulnerability <= 0f)
                {
                    playerHp = Mathf.Max(0, playerHp - 2);
                    invulnerability = 0.75f;
                }

                if (bullet.position.x < battleBox.xMin - 80 || bullet.position.x > battleBox.xMax + 80 ||
                    bullet.position.y < battleBox.yMin - 80 || bullet.position.y > battleBox.yMax + 80)
                    bullets.RemoveAt(i);
            }

            if (playerHp <= 0)
            {
                bullets.Clear();
                battlePhase = BattlePhase.Defeat;
                battleMessage = "Пикс падает. Но история ещё не обязана закончиться.";
            }
            else if (enemyTurnRemaining <= 0f)
            {
                bullets.Clear();
                battlePhase = BattlePhase.Menu;
                battleMessage = "Страж Банки ждёт решения Пикса.";
            }
        }

        void SpawnBulletPattern()
        {
            Bullet bullet = new Bullet();
            bullet.radius = 16f;

            if (attackPattern == 0)
            {
                bullet.position = new Vector2(UnityEngine.Random.Range(battleBox.xMin + 30, battleBox.xMax - 30), battleBox.yMin - 25);
                bullet.velocity = new Vector2(UnityEngine.Random.Range(-45f, 45f), UnityEngine.Random.Range(290f, 390f));
            }
            else if (attackPattern == 1)
            {
                bool left = UnityEngine.Random.value > 0.5f;
                bullet.position = new Vector2(left ? battleBox.xMin - 25 : battleBox.xMax + 25,
                    UnityEngine.Random.Range(battleBox.yMin + 30, battleBox.yMax - 30));
                bullet.velocity = new Vector2(left ? UnityEngine.Random.Range(330f, 430f) : UnityEngine.Random.Range(-430f, -330f),
                    UnityEngine.Random.Range(-55f, 55f));
            }
            else
            {
                float angle = UnityEngine.Random.Range(0f, Mathf.PI * 2f);
                Vector2 direction = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
                bullet.position = battleBox.center + direction * 430f;
                bullet.velocity = -direction * UnityEngine.Random.Range(300f, 410f);
                bullet.radius = 13f;
            }

            bullets.Add(bullet);
        }

        void StartPeacefulEnding()
        {
            mode = Mode.Ending;
            endingTitle = "ТЁПЛАЯ КОНЦОВКА";
            endingLines = new[]
            {
                "Внутри банки не было чудовища. Там прятался старый механический хранитель.",
                "Он украл семечко, потому что боялся снова остаться один в холодной кладовой.",
                "Пикс принёс свет всем — и оставил Стражу место у общего стола.",
                "Зимой в Норной Долине впервые стало теплее не только от еды."
            };
            endingIndex = 0;
            PlayerPrefs.SetInt("HamsterTalePeace", 1);
            PlayerPrefs.Save();
        }

        void StartDarkEnding()
        {
            mode = Mode.Ending;
            endingTitle = "ТИХАЯ КОНЦОВКА";
            endingLines = new[]
            {
                "Золотое семечко найдено.",
                "Но в кладовой больше никто не отвечает на эхо шагов.",
                "Долина переживает зиму. Пикс — тоже.",
                "Почему же тогда Последний Фонарь светит так холодно?"
            };
            endingIndex = 0;
            PlayerPrefs.SetInt("HamsterTaleDark", 1);
            PlayerPrefs.Save();
        }

        void OnGUI()
        {
            float scaleX = Screen.width / W;
            float scaleY = Screen.height / H;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(scaleX, scaleY, 1f));
            EnsureStyles();

            if (mode == Mode.Title) DrawTitle();
            else if (mode == Mode.Story) DrawStory();
            else if (mode == Mode.Explore) DrawExplore();
            else if (mode == Mode.Dialogue) DrawDialogue();
            else if (mode == Mode.Battle) DrawBattle();
            else if (mode == Mode.Ending) DrawEnding();
        }

        void DrawTitle()
        {
            Fill(new Rect(0, 0, W, H), ink);
            for (int i = 0; i < 12; i++)
            {
                float x = 95 + i * 165;
                Fill(new Rect(x, 120 + (i % 3) * 30, 8, 8), new Color(1f, 1f, 1f, 0.35f));
            }

            DrawSprite(hamsterSprite, new Rect(760, 175, 400, 400));
            Label(new Rect(280, 545, 1360, 105), "HAMSTER TALE", titleStyle);
            Label(new Rect(360, 650, 1200, 65), "ПОСЛЕДНЕЕ СЕМЕЧКО", headingStyle);
            DrawButton(startButton, "НАЧАТЬ ИСТОРИЮ", gold, ink);
            Label(new Rect(480, 950, 960, 50), "2D STORY RPG • TOUCH CONTROLS • TWO ENDINGS", smallStyle);
        }

        void DrawStory()
        {
            Fill(new Rect(0, 0, W, H), new Color(0.035f, 0.03f, 0.05f));
            DrawSprite(hamsterSprite, new Rect(180, 280, 420, 420));
            DrawPanel(new Rect(620, 245, 1110, 500), paper, ink, 8f);
            Label(new Rect(700, 330, 950, 270), introLines[Mathf.Clamp(storyIndex, 0, introLines.Length - 1)], bodyStyle);
            Label(new Rect(700, 610, 900, 60), (storyIndex + 1) + " / " + introLines.Length, smallStyle);
            DrawButton(nextButton, storyIndex == introLines.Length - 1 ? "ВОЙТИ" : "ДАЛЬШЕ", gold, ink);
        }

        void DrawExplore()
        {
            Fill(new Rect(0, 0, W, H), new Color(0.10f, 0.08f, 0.12f));
            DrawPanel(new Rect(150, 110, 1620, 710), new Color(0.43f, 0.31f, 0.22f), ink, 12f);

            for (int y = 150; y < 790; y += 80)
                for (int x = 190; x < 1730; x += 100)
                    Fill(new Rect(x, y, 92, 72), ((x + y) / 10) % 2 == 0 ? new Color(0.48f, 0.35f, 0.24f) : new Color(0.40f, 0.28f, 0.20f));

            Fill(new Rect(1530, 175, 170, 250), new Color(0.17f, 0.18f, 0.22f));
            DrawPanel(new Rect(1550, 195, 130, 210), new Color(0.30f, 0.34f, 0.39f), ink, 5f);
            Fill(new Rect(1650, 295, 14, 14), gold);

            Fill(new Rect(390, 270, 180, 110), new Color(0.70f, 0.58f, 0.38f));
            Fill(new Rect(420, 300, 120, 55), new Color(0.18f, 0.12f, 0.10f));

            DrawSprite(moleSprite, new Rect(865, 390, 180, 180));
            Label(new Rect(840, 565, 230, 45), "ТИХОН", smallStyle);
            DrawSprite(hamsterSprite, new Rect(playerPosition.x - 75, playerPosition.y - 120, 150, 150));

            Label(new Rect(170, 25, 1580, 65), questStep == 0
                ? "ЗАДАЧА: поговорить с Тихоном"
                : "ЗАДАЧА: открыть железную дверь", headingStyle);

            DrawDPad();
            DrawButton(actionButton, "ДЕЙСТВИЕ", gold, ink);

            string hint = "";
            if (Vector2.Distance(playerPosition, new Vector2(950, 500)) < 210) hint = "Тихон хочет что-то сказать";
            else if (Vector2.Distance(playerPosition, new Vector2(480, 350)) < 190) hint = "Пустая миска";
            else if (Vector2.Distance(playerPosition, new Vector2(1570, 290)) < 220) hint = "Железная дверь";
            if (hint.Length > 0)
            {
                DrawPanel(new Rect(650, 740, 620, 70), ink, paper, 4f);
                Label(new Rect(675, 752, 570, 45), hint, smallStyle);
            }
        }

        void DrawDialogue()
        {
            DrawExploreBackgroundOnly();
            DrawPanel(new Rect(120, 710, 1680, 290), ink, paper, 9f);
            Label(new Rect(190, 765, 1400, 150), dialogueLines[Mathf.Clamp(dialogueIndex, 0, dialogueLines.Length - 1)], bodyStyle);
            DrawButton(nextButton, dialogueIndex == dialogueLines.Length - 1 ? "ГОТОВО" : "ДАЛЬШЕ", gold, ink);
        }

        void DrawExploreBackgroundOnly()
        {
            Fill(new Rect(0, 0, W, H), new Color(0.10f, 0.08f, 0.12f));
            DrawPanel(new Rect(150, 110, 1620, 710), new Color(0.43f, 0.31f, 0.22f), ink, 12f);
            for (int y = 150; y < 790; y += 80)
                for (int x = 190; x < 1730; x += 100)
                    Fill(new Rect(x, y, 92, 72), ((x + y) / 10) % 2 == 0 ? new Color(0.48f, 0.35f, 0.24f) : new Color(0.40f, 0.28f, 0.20f));
            Fill(new Rect(1530, 175, 170, 250), new Color(0.17f, 0.18f, 0.22f));
            DrawSprite(moleSprite, new Rect(865, 390, 180, 180));
            DrawSprite(hamsterSprite, new Rect(playerPosition.x - 75, playerPosition.y - 120, 150, 150));
        }

        void DrawBattle()
        {
            Fill(new Rect(0, 0, W, H), new Color(0.035f, 0.035f, 0.055f));
            DrawSprite(guardianSprite, new Rect(780, 25, 360, 250));
            Label(new Rect(40, 35, 520, 55), "ПИКС  HP " + playerHp + "/20", headingStyle);
            Label(new Rect(1360, 35, 520, 55), "СТРАЖ  HP " + enemyHp + "/30", headingStyle);
            Label(new Rect(1360, 95, 520, 45), "ДОВЕРИЕ " + mercy + "%", smallStyle);

            DrawPanel(battleBox, new Color(0.025f, 0.02f, 0.035f), paper, 8f);

            if (battlePhase == BattlePhase.EnemyTurn)
            {
                for (int i = 0; i < bullets.Count; i++)
                {
                    Bullet bullet = bullets[i];
                    Fill(new Rect(bullet.position.x - bullet.radius, bullet.position.y - bullet.radius,
                        bullet.radius * 2, bullet.radius * 2), rose);
                }

                Color soulColor = invulnerability > 0f && Mathf.FloorToInt(invulnerability * 14f) % 2 == 0
                    ? new Color(1f, 1f, 1f, 0.25f)
                    : gold;
                GUI.color = soulColor;
                GUI.DrawTexture(new Rect(soulPosition.x - 22, soulPosition.y - 26, 44, 52), seedSprite, ScaleMode.StretchToFill, true);
                GUI.color = Color.white;
                DrawDPad();
                Label(new Rect(670, 700, 580, 50), "УКЛОНЯЙСЯ • " + Mathf.CeilToInt(enemyTurnRemaining), centerStyle);
            }
            else
            {
                DrawPanel(new Rect(545, 690, 830, 125), ink, paper, 4f);
                Label(new Rect(575, 715, 770, 80), battleMessage, bodyStyle);
            }

            if (battlePhase == BattlePhase.Menu)
            {
                DrawButton(battleTalk, "ГОВОРИТЬ", blue, Color.white);
                DrawButton(battleSpare, "ПОЩАДИТЬ", moss, Color.white);
                DrawButton(battleSnack, "КРОШКА " + snacks, gold, ink);
                DrawButton(battleAttack, "УДАР", rose, Color.white);
            }
            else if (battlePhase == BattlePhase.Message)
            {
                Label(new Rect(720, 825, 480, 45), "КОСНИСЬ, ЧТОБЫ ПРОДОЛЖИТЬ", smallStyle);
            }
            else if (battlePhase == BattlePhase.Defeat)
            {
                DrawPanel(new Rect(490, 700, 940, 110), ink, paper, 4f);
                Label(new Rect(530, 725, 860, 65), battleMessage, bodyStyle);
                DrawButton(restartButton, "ПОПРОБОВАТЬ СНОВА", gold, ink);
            }
        }

        void DrawEnding()
        {
            Fill(new Rect(0, 0, W, H), ink);
            Label(new Rect(260, 90, 1400, 90), endingTitle, titleStyle);
            DrawSprite(hamsterSprite, new Rect(190, 300, 360, 360));
            DrawPanel(new Rect(590, 260, 1130, 460), paper, ink, 8f);

            if (endingIndex < endingLines.Length)
            {
                Label(new Rect(680, 340, 950, 230), endingLines[endingIndex], bodyStyle);
                Label(new Rect(680, 610, 500, 45), (endingIndex + 1) + " / " + endingLines.Length, smallStyle);
                DrawButton(nextButton, endingIndex == endingLines.Length - 1 ? "ФИНАЛ" : "ДАЛЬШЕ", gold, ink);
            }
            else
            {
                Label(new Rect(680, 360, 950, 170), "ГЛАВА 1 ЗАВЕРШЕНА", headingStyle);
                Label(new Rect(680, 520, 950, 110), "Твой выбор сохранён на этом устройстве.", bodyStyle);
                DrawButton(restartButton, "НАЧАТЬ ЗАНОВО", gold, ink);
            }
        }

        void DrawDPad()
        {
            DrawButton(upButton, "^", new Color(0.16f, 0.16f, 0.22f, 0.94f), paper);
            DrawButton(downButton, "v", new Color(0.16f, 0.16f, 0.22f, 0.94f), paper);
            DrawButton(leftButton, "<", new Color(0.16f, 0.16f, 0.22f, 0.94f), paper);
            DrawButton(rightButton, ">", new Color(0.16f, 0.16f, 0.22f, 0.94f), paper);
        }

        Vector2 ReadMove()
        {
            Vector2 move = Vector2.zero;
            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow)) move.y -= 1f;
            if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow)) move.y += 1f;
            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)) move.x -= 1f;
            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) move.x += 1f;

            for (int i = 0; i < Input.touchCount; i++)
            {
                Vector2 p = ToLogical(Input.GetTouch(i).position);
                if (upButton.Contains(p)) move.y -= 1f;
                if (downButton.Contains(p)) move.y += 1f;
                if (leftButton.Contains(p)) move.x -= 1f;
                if (rightButton.Contains(p)) move.x += 1f;
            }

            if (Input.GetMouseButton(0))
            {
                Vector2 p = ToLogical(Input.mousePosition);
                if (upButton.Contains(p)) move.y -= 1f;
                if (downButton.Contains(p)) move.y += 1f;
                if (leftButton.Contains(p)) move.x -= 1f;
                if (rightButton.Contains(p)) move.x += 1f;
            }

            return move.sqrMagnitude > 1f ? move.normalized : move;
        }

        bool TryGetBattleDrag(out Vector2 p)
        {
            for (int i = 0; i < Input.touchCount; i++)
            {
                Touch touch = Input.GetTouch(i);
                Vector2 logical = ToLogical(touch.position);
                if (battleBox.Contains(logical))
                {
                    p = logical;
                    return true;
                }
            }

            if (Input.GetMouseButton(0))
            {
                Vector2 logical = ToLogical(Input.mousePosition);
                if (battleBox.Contains(logical))
                {
                    p = logical;
                    return true;
                }
            }

            p = Vector2.zero;
            return false;
        }

        bool TapIn(Rect rect)
        {
            for (int i = 0; i < Input.touchCount; i++)
            {
                Touch touch = Input.GetTouch(i);
                if (touch.phase == TouchPhase.Began && rect.Contains(ToLogical(touch.position)))
                    return true;
            }

            return Input.GetMouseButtonDown(0) && rect.Contains(ToLogical(Input.mousePosition));
        }

        bool AnyTap()
        {
            for (int i = 0; i < Input.touchCount; i++)
                if (Input.GetTouch(i).phase == TouchPhase.Began)
                    return true;
            return Input.GetMouseButtonDown(0) || Input.GetKeyDown(KeyCode.Space) || Input.GetKeyDown(KeyCode.Return);
        }

        Vector2 ToLogical(Vector2 screenPosition)
        {
            return new Vector2(screenPosition.x * W / Mathf.Max(1f, Screen.width),
                (Screen.height - screenPosition.y) * H / Mathf.Max(1f, Screen.height));
        }

        void EnsureStyles()
        {
            if (titleStyle != null) return;

            titleStyle = new GUIStyle(GUI.skin.label);
            titleStyle.fontSize = 76;
            titleStyle.fontStyle = FontStyle.Bold;
            titleStyle.alignment = TextAnchor.MiddleCenter;
            titleStyle.normal.textColor = paper;

            headingStyle = new GUIStyle(GUI.skin.label);
            headingStyle.fontSize = 42;
            headingStyle.fontStyle = FontStyle.Bold;
            headingStyle.alignment = TextAnchor.MiddleCenter;
            headingStyle.normal.textColor = paper;

            bodyStyle = new GUIStyle(GUI.skin.label);
            bodyStyle.fontSize = 35;
            bodyStyle.wordWrap = true;
            bodyStyle.alignment = TextAnchor.MiddleLeft;
            bodyStyle.normal.textColor = paper;

            smallStyle = new GUIStyle(GUI.skin.label);
            smallStyle.fontSize = 25;
            smallStyle.alignment = TextAnchor.MiddleCenter;
            smallStyle.normal.textColor = paper;

            buttonStyle = new GUIStyle(GUI.skin.label);
            buttonStyle.fontSize = 34;
            buttonStyle.fontStyle = FontStyle.Bold;
            buttonStyle.alignment = TextAnchor.MiddleCenter;
            buttonStyle.normal.textColor = ink;

            centerStyle = new GUIStyle(GUI.skin.label);
            centerStyle.fontSize = 30;
            centerStyle.alignment = TextAnchor.MiddleCenter;
            centerStyle.normal.textColor = paper;
        }

        void DrawButton(Rect rect, string text, Color background, Color foreground)
        {
            Fill(rect, background);
            Fill(new Rect(rect.x, rect.y, rect.width, 6), foreground);
            Fill(new Rect(rect.x, rect.yMax - 6, rect.width, 6), foreground);
            Fill(new Rect(rect.x, rect.y, 6, rect.height), foreground);
            Fill(new Rect(rect.xMax - 6, rect.y, 6, rect.height), foreground);
            Color old = buttonStyle.normal.textColor;
            buttonStyle.normal.textColor = foreground;
            Label(rect, text, buttonStyle);
            buttonStyle.normal.textColor = old;
        }

        void DrawPanel(Rect rect, Color background, Color border, float thickness)
        {
            Fill(rect, border);
            Fill(new Rect(rect.x + thickness, rect.y + thickness, rect.width - thickness * 2f, rect.height - thickness * 2f), background);
        }

        void Label(Rect rect, string text, GUIStyle style)
        {
            GUI.Label(rect, text, style);
        }

        void Fill(Rect rect, Color color)
        {
            Color old = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(rect, pixel);
            GUI.color = old;
        }

        void DrawSprite(Texture2D texture, Rect rect)
        {
            GUI.DrawTexture(rect, texture, ScaleMode.StretchToFill, true);
        }

        void CreateTextures()
        {
            pixel = SolidTexture(Color.white);
            hamsterSprite = BuildHamsterSprite();
            moleSprite = BuildMoleSprite();
            guardianSprite = BuildGuardianSprite();
            seedSprite = BuildSeedSprite();
        }

        Texture2D SolidTexture(Color color)
        {
            Texture2D texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, color);
            texture.Apply();
            texture.wrapMode = TextureWrapMode.Clamp;
            return texture;
        }

        Texture2D NewPixelTexture(int width, int height)
        {
            Texture2D texture = new Texture2D(width, height, TextureFormat.RGBA32, false);
            Color clear = new Color(0, 0, 0, 0);
            Color[] colors = new Color[width * height];
            for (int i = 0; i < colors.Length; i++) colors[i] = clear;
            texture.SetPixels(colors);
            texture.filterMode = FilterMode.Point;
            texture.wrapMode = TextureWrapMode.Clamp;
            return texture;
        }

        void Paint(Texture2D texture, int x, int y, int width, int height, Color color)
        {
            for (int py = y; py < y + height; py++)
                for (int px = x; px < x + width; px++)
                    if (px >= 0 && px < texture.width && py >= 0 && py < texture.height)
                        texture.SetPixel(px, py, color);
        }

        Texture2D BuildHamsterSprite()
        {
            Texture2D t = NewPixelTexture(32, 32);
            Color fur = new Color(0.72f, 0.42f, 0.20f);
            Color cream = new Color(0.95f, 0.78f, 0.50f);
            Color dark = new Color(0.06f, 0.04f, 0.05f);
            Color pink = new Color(0.96f, 0.43f, 0.50f);

            Paint(t, 8, 4, 16, 20, fur);
            Paint(t, 6, 9, 20, 11, fur);
            Paint(t, 10, 2, 4, 5, pink);
            Paint(t, 18, 2, 4, 5, pink);
            Paint(t, 11, 8, 10, 9, cream);
            Paint(t, 10, 11, 3, 3, dark);
            Paint(t, 19, 11, 3, 3, dark);
            Paint(t, 15, 8, 2, 2, pink);
            Paint(t, 11, 20, 10, 8, cream);
            Paint(t, 7, 25, 7, 4, cream);
            Paint(t, 18, 25, 7, 4, cream);
            t.Apply();
            return t;
        }

        Texture2D BuildMoleSprite()
        {
            Texture2D t = NewPixelTexture(32, 32);
            Color coat = new Color(0.28f, 0.24f, 0.30f);
            Color face = new Color(0.65f, 0.50f, 0.37f);
            Color glass = new Color(0.45f, 0.78f, 0.85f);
            Color dark = new Color(0.04f, 0.04f, 0.05f);

            Paint(t, 8, 4, 16, 23, coat);
            Paint(t, 10, 8, 12, 11, face);
            Paint(t, 8, 10, 7, 6, glass);
            Paint(t, 17, 10, 7, 6, glass);
            Paint(t, 15, 12, 2, 2, dark);
            Paint(t, 12, 18, 8, 8, new Color(0.18f, 0.16f, 0.20f));
            t.Apply();
            return t;
        }

        Texture2D BuildGuardianSprite()
        {
            Texture2D t = NewPixelTexture(48, 32);
            Color metal = new Color(0.42f, 0.48f, 0.53f);
            Color edge = new Color(0.18f, 0.20f, 0.24f);
            Color eye = new Color(1f, 0.56f, 0.18f);

            Paint(t, 10, 3, 28, 25, metal);
            Paint(t, 7, 5, 34, 4, edge);
            Paint(t, 7, 24, 34, 4, edge);
            Paint(t, 14, 11, 7, 6, edge);
            Paint(t, 27, 11, 7, 6, edge);
            Paint(t, 16, 12, 3, 3, eye);
            Paint(t, 29, 12, 3, 3, eye);
            Paint(t, 18, 20, 12, 3, edge);
            Paint(t, 3, 12, 7, 6, metal);
            Paint(t, 38, 12, 7, 6, metal);
            t.Apply();
            return t;
        }

        Texture2D BuildSeedSprite()
        {
            Texture2D t = NewPixelTexture(16, 20);
            Color bright = new Color(1f, 0.78f, 0.20f);
            Color shade = new Color(0.82f, 0.44f, 0.10f);
            Paint(t, 6, 1, 4, 3, bright);
            Paint(t, 4, 4, 8, 11, bright);
            Paint(t, 6, 15, 4, 4, shade);
            Paint(t, 4, 9, 3, 6, shade);
            t.Apply();
            return t;
        }
    }
}
