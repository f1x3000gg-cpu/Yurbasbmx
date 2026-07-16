package com.dippix.hamsterflip;

import android.app.Activity;
import android.graphics.Color;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.TextView;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

public final class MainActivity extends Activity {
    private GameView game;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        FrameLayout root = new FrameLayout(this);
        game = new GameView(this);
        root.addView(game, new FrameLayout.LayoutParams(-1, -1));

        TextView title = new TextView(this);
        title.setText("HAMSTER FLIP");
        title.setTextColor(Color.WHITE);
        title.setTextSize(22);
        title.setGravity(Gravity.CENTER);
        title.setShadowLayer(5f, 0f, 2f, Color.BLACK);
        FrameLayout.LayoutParams tp = new FrameLayout.LayoutParams(dp(270), dp(58));
        tp.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
        tp.topMargin = dp(18);
        root.addView(title, tp);

        Button flip = new Button(this);
        flip.setText("FLIP!");
        flip.setTextColor(Color.WHITE);
        flip.setTextSize(26);
        flip.setBackgroundColor(Color.rgb(230, 102, 35));
        flip.setOnClickListener(v -> game.flip());
        FrameLayout.LayoutParams bp = new FrameLayout.LayoutParams(dp(185), dp(96));
        bp.gravity = Gravity.BOTTOM | Gravity.RIGHT;
        bp.rightMargin = dp(34);
        bp.bottomMargin = dp(30);
        root.addView(flip, bp);
        setContentView(root);
    }

    private int dp(int n) { return Math.round(n * getResources().getDisplayMetrics().density); }
    @Override protected void onResume() { super.onResume(); game.onResume(); }
    @Override protected void onPause() { game.onPause(); super.onPause(); }

    static final class GameView extends GLSurfaceView {
        final Renderer3D renderer = new Renderer3D();
        GameView(Activity context) {
            super(context);
            setEGLContextClientVersion(2);
            setRenderer(renderer);
            setRenderMode(RENDERMODE_CONTINUOUSLY);
        }
        void flip() { queueEvent(renderer::requestFlip); }
    }

    static final class Renderer3D implements GLSurfaceView.Renderer {
        static final float G = -9.81f, BOUNCE = 7.45f, FLIP_TIME = 0.56f;
        final float[] projection = new float[16], view = new float[16], vp = new float[16];
        final float[] root = new float[16], local = new float[16], model = new float[16], mvp = new float[16];
        Sphere sphere;
        int program, aPos, aNormal, uMvp, uModel, uColor;
        long last;
        float height = 0.05f, velocity = BOUNCE, flipClock, squash;
        boolean queued, flipping;

        @Override public void onSurfaceCreated(javax.microedition.khronos.egl.EGLConfig cfg) {
            GLES20.glClearColor(0.045f, 0.12f, 0.17f, 1f);
            GLES20.glEnable(GLES20.GL_DEPTH_TEST);
            GLES20.glEnable(GLES20.GL_CULL_FACE);
            sphere = new Sphere(18, 28);
            program = program(VS, FS);
            aPos = GLES20.glGetAttribLocation(program, "aPosition");
            aNormal = GLES20.glGetAttribLocation(program, "aNormal");
            uMvp = GLES20.glGetUniformLocation(program, "uMvp");
            uModel = GLES20.glGetUniformLocation(program, "uModel");
            uColor = GLES20.glGetUniformLocation(program, "uColor");
            last = System.nanoTime();
        }

        @Override public void onSurfaceChanged(javax.microedition.khronos.opengles.GL10 gl, int w, int h) {
            GLES20.glViewport(0, 0, w, h);
            Matrix.perspectiveM(projection, 0, 47f, w / (float)Math.max(1, h), 0.1f, 100f);
            Matrix.setLookAtM(view, 0, 0f, 3.25f, 8.5f, 0f, 1.7f, 0f, 0f, 1f, 0f);
            Matrix.multiplyMM(vp, 0, projection, 0, view, 0);
        }

        @Override public void onDrawFrame(javax.microedition.khronos.opengles.GL10 gl) {
            long now = System.nanoTime();
            float dt = Math.min(0.033f, Math.max(0.001f, (now - last) / 1_000_000_000f));
            last = now;
            update(dt);
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
            GLES20.glUseProgram(program);
            world();
            hamster();
        }

        void requestFlip() { queued = true; }

        void update(float dt) {
            velocity += G * dt;
            height += velocity * dt;
            if (height <= 0f && velocity < 0f) {
                height = 0f;
                velocity = BOUNCE;
                squash = 1f;
            }
            squash = Math.max(0f, squash - dt * 4.5f);
            if (queued && height > 0.18f && !flipping) {
                queued = false;
                flipping = true;
                flipClock = 0f;
            }
            if (flipping) {
                flipClock += dt;
                if (flipClock >= FLIP_TIME) {
                    flipClock = FLIP_TIME;
                    flipping = false;
                }
            }
        }

        void world() {
            draw(null, 0f,-0.72f,0f, 7.5f,0.28f,7.5f, .13f,.23f,.25f);
            draw(null, 0f,0.02f,0f, 2.25f,.10f,2.25f, .10f,.32f,.52f);
            for (int i=0;i<28;i++) {
                double a=i*Math.PI*2.0/28.0;
                draw(null,(float)Math.cos(a)*2.26f,.12f,(float)Math.sin(a)*2.26f,.18f,.13f,.18f,.95f,.42f,.08f);
            }
            leg(-1.65f,-1.65f); leg(1.65f,-1.65f); leg(-1.65f,1.65f); leg(1.65f,1.65f);
        }

        void leg(float x,float z) { draw(null,x,-.35f,z,.12f,.62f,.12f,.76f,.78f,.80f); }

        void hamster() {
            float t = flipping ? Math.min(1f, flipClock / FLIP_TIME) : 0f;
            float angle = flipping ? (t*t*(3f-2f*t))*360f : 0f;
            Matrix.setIdentityM(root,0);
            Matrix.translateM(root,0,0f,1.15f+height,0f);
            Matrix.rotateM(root,0,angle,1f,0f,0f);
            Matrix.scaleM(root,0,1f+squash*.10f,1f-squash*.16f,1f+squash*.10f);

            part(0,0,0,.72f,.86f,.56f,.73f,.43f,.20f);
            part(0,.69f,.04f,.58f,.54f,.53f,.73f,.43f,.20f);
            part(0,.59f,.48f,.34f,.26f,.19f,.96f,.76f,.48f);
            part(0,.54f,.66f,.12f,.10f,.10f,.95f,.45f,.50f);
            part(-.36f,1.05f,.03f,.20f,.25f,.15f,.73f,.43f,.20f);
            part(.36f,1.05f,.03f,.20f,.25f,.15f,.73f,.43f,.20f);
            part(-.36f,1.06f,.06f,.11f,.14f,.09f,.95f,.45f,.50f);
            part(.36f,1.06f,.06f,.11f,.14f,.09f,.95f,.45f,.50f);
            part(-.23f,.78f,.48f,.07f,.08f,.06f,.04f,.03f,.03f);
            part(.23f,.78f,.48f,.07f,.08f,.06f,.04f,.03f,.03f);
            part(-.61f,.04f,.02f,.18f,.48f,.18f,.73f,.43f,.20f);
            part(.61f,.04f,.02f,.18f,.48f,.18f,.73f,.43f,.20f);
            part(-.35f,-.70f,.16f,.28f,.17f,.38f,.96f,.76f,.48f);
            part(.35f,-.70f,.16f,.28f,.17f,.38f,.96f,.76f,.48f);
            part(0,-.03f,-.50f,.30f,.30f,.26f,.73f,.43f,.20f);
        }

        void part(float x,float y,float z,float sx,float sy,float sz,float r,float g,float b) {
            draw(root,x,y,z,sx,sy,sz,r,g,b);
        }

        void draw(float[] parent,float x,float y,float z,float sx,float sy,float sz,float r,float g,float b) {
            Matrix.setIdentityM(local,0);
            Matrix.translateM(local,0,x,y,z);
            Matrix.scaleM(local,0,sx,sy,sz);
            if (parent==null) System.arraycopy(local,0,model,0,16);
            else Matrix.multiplyMM(model,0,parent,0,local,0);
            Matrix.multiplyMM(mvp,0,vp,0,model,0);
            GLES20.glUniformMatrix4fv(uMvp,1,false,mvp,0);
            GLES20.glUniformMatrix4fv(uModel,1,false,model,0);
            GLES20.glUniform4f(uColor,r,g,b,1f);
            sphere.draw(aPos,aNormal);
        }

        static int program(String vs,String fs) {
            int v=shader(GLES20.GL_VERTEX_SHADER,vs), f=shader(GLES20.GL_FRAGMENT_SHADER,fs);
            int p=GLES20.glCreateProgram();
            GLES20.glAttachShader(p,v); GLES20.glAttachShader(p,f); GLES20.glLinkProgram(p);
            int[] ok=new int[1]; GLES20.glGetProgramiv(p,GLES20.GL_LINK_STATUS,ok,0);
            if(ok[0]==0) throw new IllegalStateException(GLES20.glGetProgramInfoLog(p));
            GLES20.glDeleteShader(v); GLES20.glDeleteShader(f); return p;
        }

        static int shader(int type,String src) {
            int s=GLES20.glCreateShader(type); GLES20.glShaderSource(s,src); GLES20.glCompileShader(s);
            int[] ok=new int[1]; GLES20.glGetShaderiv(s,GLES20.GL_COMPILE_STATUS,ok,0);
            if(ok[0]==0) throw new IllegalStateException(GLES20.glGetShaderInfoLog(s));
            return s;
        }

        static final String VS="uniform mat4 uMvp;uniform mat4 uModel;attribute vec3 aPosition;attribute vec3 aNormal;varying float vLight;void main(){vec3 n=normalize(mat3(uModel)*aNormal);vec3 l=normalize(vec3(-.35,.85,.45));vLight=max(dot(n,l),0.0);gl_Position=uMvp*vec4(aPosition,1.0);}";
        static final String FS="precision mediump float;uniform vec4 uColor;varying float vLight;void main(){float l=.34+vLight*.66;gl_FragColor=vec4(uColor.rgb*l,uColor.a);}";

        static final class Sphere {
            final FloatBuffer vertices,normals; final ShortBuffer indices; final int count;
            Sphere(int stacks,int slices) {
                int vc=(stacks+1)*(slices+1), p=0, q=0;
                float[] v=new float[vc*3], n=new float[vc*3];
                short[] ix=new short[stacks*slices*6];
                for(int i=0;i<=stacks;i++) {
                    float phi=(float)Math.PI*i/stacks, y=(float)Math.cos(phi), ring=(float)Math.sin(phi);
                    for(int j=0;j<=slices;j++) {
                        float th=(float)(Math.PI*2.0*j/slices);
                        float x=ring*(float)Math.cos(th), z=ring*(float)Math.sin(th);
                        v[p]=x;n[p++]=x; v[p]=y;n[p++]=y; v[p]=z;n[p++]=z;
                    }
                }
                for(int i=0;i<stacks;i++) for(int j=0;j<slices;j++) {
                    short a=(short)(i*(slices+1)+j), b=(short)(a+slices+1), c=(short)(a+1), d=(short)(b+1);
                    ix[q++]=a;ix[q++]=b;ix[q++]=c; ix[q++]=c;ix[q++]=b;ix[q++]=d;
                }
                vertices=floats(v); normals=floats(n);
                indices=ByteBuffer.allocateDirect(ix.length*2).order(ByteOrder.nativeOrder()).asShortBuffer();
                indices.put(ix).position(0); count=ix.length;
            }
            void draw(int pos,int normal) {
                vertices.position(0); normals.position(0); indices.position(0);
                GLES20.glEnableVertexAttribArray(pos); GLES20.glVertexAttribPointer(pos,3,GLES20.GL_FLOAT,false,12,vertices);
                GLES20.glEnableVertexAttribArray(normal); GLES20.glVertexAttribPointer(normal,3,GLES20.GL_FLOAT,false,12,normals);
                GLES20.glDrawElements(GLES20.GL_TRIANGLES,count,GLES20.GL_UNSIGNED_SHORT,indices);
                GLES20.glDisableVertexAttribArray(pos); GLES20.glDisableVertexAttribArray(normal);
            }
            static FloatBuffer floats(float[] data) {
                FloatBuffer b=ByteBuffer.allocateDirect(data.length*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
                b.put(data).position(0); return b;
            }
        }
    }
}
