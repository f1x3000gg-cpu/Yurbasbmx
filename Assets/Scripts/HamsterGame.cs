using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace HamsterFlip
{
    public sealed class HamsterBootstrap : MonoBehaviour
    {
        Font font;

        void Awake()
        {
            Application.targetFrameRate = 60;
            QualitySettings.vSyncCount = 0;
            Time.fixedDeltaTime = 0.02f;
            Screen.orientation = ScreenOrientation.LandscapeLeft;
            font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            BuildLight();
            BuildGame();
        }

        void BuildLight()
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(.55f,.68f,.82f);
            RenderSettings.ambientEquatorColor = new Color(.32f,.40f,.46f);
            RenderSettings.ambientGroundColor = new Color(.1f,.11f,.13f);
            GameObject go = new GameObject("Sun");
            Light l = go.AddComponent<Light>();
            l.type = LightType.Directional;
            l.intensity = 1.2f;
            l.shadows = LightShadows.Soft;
            go.transform.rotation = Quaternion.Euler(48f,-32f,0f);
        }

        void BuildGame()
        {
            Primitive(PrimitiveType.Cube,"Ground",new Vector3(0,-.35f,0),new Vector3(28,.7f,28),new Color(.24f,.44f,.22f),null);
            BuildTrampoline();
            HamsterController h = BuildHamster();
            BuildCamera(h.transform);
            BuildUI(h);
        }

        void BuildTrampoline()
        {
            GameObject root = new GameObject("Trampoline");
            GameObject mat = Primitive(PrimitiveType.Cylinder,"BounceMat",new Vector3(0,.78f,0),new Vector3(2.5f,.09f,2.5f),new Color(.08f,.1f,.13f),root.transform);
            mat.tag = "Respawn";
            Destroy(mat.GetComponent<Collider>());
            BoxCollider box = mat.AddComponent<BoxCollider>();
            box.size = new Vector3(1f,2f,1f);

            Material red = Make(new Color(.86f,.2f,.15f));
            const int count = 28;
            for (int i=0;i<count;i++)
            {
                float a=i*Mathf.PI*2f/count, b=(i+1)*Mathf.PI*2f/count;
                Vector3 p1=new Vector3(Mathf.Cos(a)*2.65f,.84f,Mathf.Sin(a)*2.65f);
                Vector3 p2=new Vector3(Mathf.Cos(b)*2.65f,.84f,Mathf.Sin(b)*2.65f);
                GameObject s=GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                s.name="Ring"; s.transform.SetParent(root.transform,false);
                s.transform.localPosition=(p1+p2)*.5f;
                s.transform.localScale=new Vector3(.09f,Vector3.Distance(p1,p2)*.5f,.09f);
                s.transform.localRotation=Quaternion.FromToRotation(Vector3.up,p2-p1);
                s.GetComponent<Renderer>().material=red;
                Destroy(s.GetComponent<Collider>());
            }
            for(int i=0;i<8;i++)
            {
                float a=i*Mathf.PI*2f/8f;
                Primitive(PrimitiveType.Cylinder,"Leg",new Vector3(Mathf.Cos(a)*2.45f,.36f,Mathf.Sin(a)*2.45f),new Vector3(.09f,.42f,.09f),new Color(.15f,.16f,.18f),root.transform,true);
            }
        }

        HamsterController BuildHamster()
        {
            GameObject root=new GameObject("Hamster");
            root.transform.position=new Vector3(0,2f,0);
            Rigidbody rb=root.AddComponent<Rigidbody>();
            rb.mass=1.8f; rb.drag=.05f; rb.angularDrag=.15f;
            rb.interpolation=RigidbodyInterpolation.Interpolate;
            rb.collisionDetectionMode=CollisionDetectionMode.ContinuousDynamic;
            rb.constraints=RigidbodyConstraints.FreezeRotation;
            CapsuleCollider c=root.AddComponent<CapsuleCollider>();
            c.radius=.46f; c.height=1.1f; c.center=new Vector3(0,.54f,0);

            GameObject pivot=new GameObject("VisualRoot");
            pivot.transform.SetParent(root.transform,false);
            pivot.transform.localPosition=new Vector3(0,.85f,0);
            GameObject model=new GameObject("Model");
            model.transform.SetParent(pivot.transform,false);
            model.transform.localPosition=new Vector3(0,-.85f,0);
            BuildHamsterVisual(model.transform);

            HamsterController h=root.AddComponent<HamsterController>();
            h.visual=pivot.transform;
            return h;
        }

        void BuildHamsterVisual(Transform p)
        {
            Color fur=new Color(.72f,.43f,.22f), cream=new Color(.93f,.76f,.52f), dark=new Color(.03f,.02f,.015f), pink=new Color(.95f,.45f,.5f);
            Primitive(PrimitiveType.Sphere,"Body",new Vector3(0,.58f,0),new Vector3(1.05f,1.22f,.92f),fur,p,true);
            Primitive(PrimitiveType.Sphere,"Belly",new Vector3(0,.55f,-.38f),new Vector3(.7f,.86f,.18f),cream,p,true);
            Primitive(PrimitiveType.Sphere,"Head",new Vector3(0,1.32f,-.06f),new Vector3(.92f,.86f,.86f),fur,p,true);
            Primitive(PrimitiveType.Sphere,"Muzzle",new Vector3(0,1.17f,-.47f),new Vector3(.58f,.36f,.28f),cream,p,true);
            Primitive(PrimitiveType.Sphere,"EyeL",new Vector3(-.22f,1.48f,-.42f),Vector3.one*.13f,dark,p,true);
            Primitive(PrimitiveType.Sphere,"EyeR",new Vector3(.22f,1.48f,-.42f),Vector3.one*.13f,dark,p,true);
            Primitive(PrimitiveType.Sphere,"Nose",new Vector3(0,1.18f,-.64f),Vector3.one*.12f,pink,p,true);
            Primitive(PrimitiveType.Sphere,"EarL",new Vector3(-.36f,1.72f,-.02f),new Vector3(.3f,.34f,.18f),pink,p,true);
            Primitive(PrimitiveType.Sphere,"EarR",new Vector3(.36f,1.72f,-.02f),new Vector3(.3f,.34f,.18f),pink,p,true);
            Primitive(PrimitiveType.Sphere,"FootL",new Vector3(-.3f,.03f,-.18f),new Vector3(.38f,.22f,.54f),cream,p,true);
            Primitive(PrimitiveType.Sphere,"FootR",new Vector3(.3f,.03f,-.18f),new Vector3(.38f,.22f,.54f),cream,p,true);
            Primitive(PrimitiveType.Sphere,"Tail",new Vector3(0,.6f,.46f),Vector3.one*.28f,cream,p,true);
        }

        void BuildCamera(Transform target)
        {
            GameObject go=new GameObject("Main Camera"); go.tag="MainCamera";
            Camera c=go.AddComponent<Camera>(); c.fieldOfView=58f; c.nearClipPlane=.05f; c.farClipPlane=150f;
            c.clearFlags=CameraClearFlags.SolidColor; c.backgroundColor=new Color(.48f,.72f,.92f);
            go.AddComponent<AudioListener>();
            FollowCamera f=go.AddComponent<FollowCamera>(); f.target=target;
            go.transform.position=new Vector3(0,4.2f,-8.8f);
        }

        void BuildUI(HamsterController h)
        {
            if(FindObjectOfType<EventSystem>()==null)
            {
                GameObject e=new GameObject("EventSystem"); e.AddComponent<EventSystem>(); e.AddComponent<StandaloneInputModule>();
            }
            GameObject co=new GameObject("Canvas",typeof(RectTransform),typeof(Canvas),typeof(CanvasScaler),typeof(GraphicRaycaster));
            co.GetComponent<Canvas>().renderMode=RenderMode.ScreenSpaceOverlay;
            CanvasScaler s=co.GetComponent<CanvasScaler>(); s.uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize; s.referenceResolution=new Vector2(1920,1080); s.matchWidthOrHeight=.5f;
            Text title=Text(co.transform,"HAMSTER FLIP TEST",52,TextAnchor.UpperCenter); SetRect(title.rectTransform,new Vector2(.5f,1),new Vector2(.5f,1),new Vector2(0,-35),new Vector2(900,80),new Vector2(.5f,1));
            Text info=Text(co.transform,"Hamster jumps automatically. Tap FLIP in the air.",30,TextAnchor.UpperCenter); SetRect(info.rectTransform,new Vector2(.5f,1),new Vector2(.5f,1),new Vector2(0,-105),new Vector2(1100,60),new Vector2(.5f,1));
            h.score=Text(co.transform,"FLIPS: 0",38,TextAnchor.UpperLeft); SetRect(h.score.rectTransform,new Vector2(0,1),new Vector2(0,1),new Vector2(42,-42),new Vector2(420,70),new Vector2(0,1));

            GameObject bo=new GameObject("FlipButton",typeof(RectTransform),typeof(Image),typeof(Button)); bo.transform.SetParent(co.transform,false);
            RectTransform br=bo.GetComponent<RectTransform>(); SetRect(br,new Vector2(1,0),new Vector2(1,0),new Vector2(-70,70),new Vector2(340,340),new Vector2(1,0));
            Image image=bo.GetComponent<Image>(); image.color=new Color(.95f,.3f,.16f,.92f);
            Button button=bo.GetComponent<Button>(); button.targetGraphic=image; button.onClick.AddListener(h.RequestFlip);
            Text bt=Text(bo.transform,"FLIP",72,TextAnchor.MiddleCenter); bt.rectTransform.anchorMin=Vector2.zero; bt.rectTransform.anchorMax=Vector2.one; bt.rectTransform.offsetMin=Vector2.zero; bt.rectTransform.offsetMax=Vector2.zero;
        }

        Text Text(Transform p,string value,int size,TextAnchor align)
        {
            GameObject o=new GameObject("Text",typeof(RectTransform),typeof(Text)); o.transform.SetParent(p,false);
            Text t=o.GetComponent<Text>(); t.font=font; t.text=value; t.fontSize=size; t.alignment=align; t.color=Color.white;
            t.horizontalOverflow=HorizontalWrapMode.Overflow; t.verticalOverflow=VerticalWrapMode.Overflow; return t;
        }

        static void SetRect(RectTransform r,Vector2 min,Vector2 max,Vector2 pos,Vector2 size,Vector2 pivot){r.anchorMin=min;r.anchorMax=max;r.anchoredPosition=pos;r.sizeDelta=size;r.pivot=pivot;}
        static Material Make(Color c){Material m=new Material(Shader.Find("Standard"));m.color=c;m.SetFloat("_Glossiness",.25f);return m;}
        static GameObject Primitive(PrimitiveType type,string name,Vector3 pos,Vector3 scale,Color color,Transform parent,bool noCollider)
        {
            GameObject o=GameObject.CreatePrimitive(type);o.name=name;if(parent)o.transform.SetParent(parent,false);o.transform.localPosition=pos;o.transform.localScale=scale;o.GetComponent<Renderer>().material=Make(color);if(noCollider)Destroy(o.GetComponent<Collider>());return o;
        }
        static GameObject Primitive(PrimitiveType type,string name,Vector3 pos,Vector3 scale,Color color,Transform parent){return Primitive(type,name,pos,scale,color,parent,false);}
    }

    public sealed class HamsterController:MonoBehaviour
    {
        public Transform visual; public Text score;
        Rigidbody rb; bool requested,flipping; float angle,nextBounce; int flips; Vector3 spawn;
        const float Bounce=9.25f,FlipSpeed=520f;
        void Awake(){rb=GetComponent<Rigidbody>();spawn=transform.position;}
        public void RequestFlip(){requested=true;}
        void FixedUpdate()
        {
            Vector3 v=rb.velocity;v.x=Mathf.MoveTowards(v.x,0,8*Time.fixedDeltaTime);v.z=Mathf.MoveTowards(v.z,0,8*Time.fixedDeltaTime);rb.velocity=v;
            bool air=transform.position.y>1.45f||rb.velocity.y>.25f;
            if(requested&&air&&!flipping){requested=false;flipping=true;angle=0;}
            if(flipping)
            {
                float step=Mathf.Min(FlipSpeed*Time.fixedDeltaTime,360-angle);angle+=step;visual.localRotation=Quaternion.Euler(angle,0,0);
                if(angle>=359.9f){flipping=false;angle=0;visual.localRotation=Quaternion.identity;flips++;if(score)score.text="FLIPS: "+flips;}
            }
            if(transform.position.y<-3){transform.position=spawn;rb.velocity=Vector3.zero;visual.localRotation=Quaternion.identity;flipping=false;angle=0;}
        }
        void OnCollisionStay(Collision c)
        {
            if(!c.gameObject.CompareTag("Respawn")||Time.time<nextBounce||rb.velocity.y>1)return;
            Vector3 v=rb.velocity;v.y=Bounce;v.x=0;v.z=0;rb.velocity=v;nextBounce=Time.time+.28f;
        }
    }

    public sealed class FollowCamera:MonoBehaviour
    {
        public Transform target; Vector3 velocity;
        void LateUpdate(){if(!target)return;Vector3 p=target.position+new Vector3(0,3.3f,-8.6f);transform.position=Vector3.SmoothDamp(transform.position,p,ref velocity,.18f);Vector3 look=target.position+Vector3.up*.8f;transform.rotation=Quaternion.Slerp(transform.rotation,Quaternion.LookRotation(look-transform.position),10*Time.deltaTime);}
    }
}
