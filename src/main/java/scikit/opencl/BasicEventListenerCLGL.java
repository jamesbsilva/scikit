package scikit.opencl;

/**
*    @(#)   BasicEventListenerCLGL
*/  

import com.jogamp.opengl.util.Animator;
import javax.media.opengl.DebugGL2;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.glu.gl2.GLUgl2;

import static com.jogamp.common.nio.Buffers.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
*       BasicEventListenerCLGL is the basic GLEventListener which is used
*   for OpenCL/OpenGL interoperability.
* 
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2013-07
*/
public class BasicEventListenerCLGL implements GLEventListener {
    private final GLUgl2 glu = new GLUgl2();
    private int width = 600;
    private int height = 400;
    private int posBuffSize = 0; private int colBuffSize = 0;
    private final int[] glObjectsCol = new int[2]; private final int[] glObjectsPos = new int[2];
    private float pointSize = 1.0f;
    private final int VERTICES = 0;
    private AtomicBoolean updateSwapInt = new AtomicBoolean(false); private int swapInterval = 20;
    private CLScheduler clscheduler;
    private CLHelper clhelper;
    private AtomicBoolean initializedGLCL = new AtomicBoolean(false);
    private SceneCL scene;
    
    /**
    *         BasicEventListenerCLGL constructor
    * 
    *  @param pbuffsize - size of position buffer 
    *  @param cbuffsize - size of color buffer 
    *  @param sin - scene being updated with this listener
    *  @param w - width
    *  @param h - height
    */ 
    public BasicEventListenerCLGL(int pbuffsize, int cbuffsize, SceneCL sin, int w, int h ) {
        this(null,pbuffsize,cbuffsize,sin,w,h);
    }
    /**
    *         BasicEventListenerCLGL constructor
    * 
    *  @param clin - clhelper to use for this GLEvent listener
    *  @param pbuffsize - size of position buffer 
    *  @param cbuffsize - size of color buffer 
    *  @param sin - scene being updated with this listener
    *  @param w - width
    *  @param h - height
    */ 
    public BasicEventListenerCLGL(CLHelper clin, int pbuffsize, int cbuffsize, SceneCL sin, int w, int h ) {
        posBuffSize = pbuffsize;
        colBuffSize = cbuffsize;
        width = w;
        height = h;
        if(clin != null){if(clin.isGLenabled()){clhelper = clin;}}
        scene = sin;
        clscheduler = new CLScheduler(this);
        pointSize = pointSizeFunction(posBuffSize);
    }
    /**
    *         getSceneKernel gets the name of the kernel which was used to initiate
    *   the position and color buffers and initially draw the scene being managed 
    *   this listener.
    * 
    *  @return kernelname of the scene initiating kernel
    */ 
    public String getSceneKernel(){
        return scene.getInitDrawKernelName();
    }
    /**
    *         getSceneKernelPosBuffNum gets the argument number of the float buffer used for the 
    *   position buffer in the kernel which was used to initiate
    *   the position and color buffers and initially draw the scene being managed 
    *   this listener.
    * 
    *  @return buffer number of float buffer for position in initiating kernel of the scene 
    */ 
    public int getSceneKernelPosBuffNum(){
        return scene.getInitDrawPosBuffNum();
    }
    /**
    *         getSceneKernelPosBuffNum gets the argument number of the float buffer used for the 
    *   color buffer in the kernel which was used to initiate
    *   the position and color buffers and initially draw the scene being managed 
    *   this listener.
    * 
    *  @return buffer number of float buffer for color in initiating kernel of the scene 
    */ 
    public int getSceneKernelColBuffNum(){
        return scene.getInitDrawColBuffNum();
    }    
    /**
    *         init is the callback for initialization of JOGL/JOCL. This method
    *   calls on the SceneCL method initCLGL where scene OpenCL should take place.
    * 
    *  @param drawable - drawable
    */ 
    @Override
    public void init(GLAutoDrawable drawable) {
        if(!initializedGLCL.get()){
            // enable GL error checking using the composable pipeline
            drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));
            // OpenGL initialization
            GL2 gl = drawable.getGL().getGL2();
            gl.setSwapInterval(swapInterval);
            gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_LINE);
            gl.glClearColor(0.10f, 0.10f, 0.10f, 0.0f);

            if(posBuffSize > 0){
                gl.glGenBuffers(glObjectsPos.length, glObjectsPos, 0);
            }
            if(colBuffSize > 0){
                gl.glGenBuffers(glObjectsCol.length, glObjectsCol, 0);
            }
            gl.glDisableClientState(GL2.GL_NORMAL_ARRAY);
            if(posBuffSize > 0){
                gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
                gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, glObjectsPos[VERTICES]);
                gl.glBufferData(GL2.GL_ARRAY_BUFFER, posBuffSize* SIZEOF_FLOAT, null, GL2.GL_DYNAMIC_DRAW);
                gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, 0);
                gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);
            }
            if(colBuffSize > 0){
                gl.glEnableClientState(GL2.GL_COLOR_ARRAY);
                gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, glObjectsCol[VERTICES]);
                gl.glBufferData(GL2.GL_ARRAY_BUFFER, colBuffSize* SIZEOF_FLOAT, null, GL2.GL_DYNAMIC_DRAW);
                gl.glDisableClientState(GL2.GL_COLOR_ARRAY);
            }
            pushPerspectiveView(gl);
            gl.glFinish();
            // init OpenCL
            initCL(drawable);
            // start rendering thread
            Animator animator = new Animator(drawable);
            animator.start();
        }
     }
    /**
    *         getCLGLInitStatus returns true if CLHelper is properly
    *   setup for interoperability. Any calls with CLHelper on a CLGL context
    *   before this will not cause good behavior.
    * 
    *  @return true - if CLGL is properly initiated
    */ 
    public boolean getCLGLInitStatus(){
        return initializedGLCL.get();
    }
    /**
    *         getPosGLind returns the index of the position buffer to be used in generating a shared OpenCL/OpenGL buffer.
    *   It is the index used in the API call (gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, index).
        
    * 
    *  @return glindex
    */ 
    public int getPosGLind(){
        return glObjectsPos[VERTICES];
    }
    /**
    *         getPosGLind returns the index of the color buffer to be used in generating a shared OpenCL/OpenGL buffer.
    *   It is the index used in the API call (gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, index).
        
    * 
    *  @return glindex
    */ 
    public int getColGLind(){
        return glObjectsCol[VERTICES];
    }
    // initialization method for OpenCL
    private void initCL(GLAutoDrawable drawable) {
        if(clhelper == null){
            scene.initCLGL(drawable);
            initializedGLCL.set(true);
        }else{
            scene.clhelper = clhelper;
            scene.initCLGL(drawable);
            initializedGLCL.set(true);
        }
    }
    /**
    *         setCLHelper sets the clhelper to be linked to this GLEventListener class
    *   which can be used for OpenCL calls which do not need to be scheduled because
    *   they are timed with an OpenGL callback function.This calls synchronizes the clhelper
    *   with the clscheduler class clhelper.
    * 
    *  @param clin - clhelper to be used in the  context for this GLEventListener
    */ 
    public void setCLhelper(CLHelper clin){
        clhelper = clin;
        clscheduler.setCLHelper(clin);
    }
    /**
    *         getCLScheduler returns the clscheduler linked to this GLEventListener class
    *   which can be used for OpenCL calls which do need to be scheduled because
    *   they are timed with an OpenGL callback function like the display method which
    *   checks for update to the display of OpenGL component.
    * 
    *  @return clscheduler with proper context for this GLEventListener
    */ 
    public CLScheduler getCLScheduler(){
        return clscheduler;
    }
    // Default point size function
    private float pointSizeFunction(int points){
        return (float)(256.0f*512.0f/(Math.pow(points,1)+2056.0f));
    }
    
    /**
    *         display is the callback function for GLevent listener
    * 
    *  @param drawable - drawable
    */ 
    @Override
    public void display(GLAutoDrawable drawable) {
        GL2 gl = drawable.getGL().getGL2();
        // ensure pipeline is clean before doing cl work
        gl.glFinish();
        // change swap interva if necessary
        if(updateSwapInt.get()){
            gl.setSwapInterval(swapInterval);
            updateSwapInt.set(false);
        }
        int mesh = (int)(Math.sqrt(posBuffSize/4));
        // kernel updates new shape
        if(initializedGLCL.get()){
            clscheduler.checkAndRun(drawable);
        }
        // Clears canvas for new shape
        gl.glClear(GL2.GL_COLOR_BUFFER_BIT | GL2.GL_DEPTH_BUFFER_BIT);
        // loads up basic
        gl.glLoadIdentity();
        //render the particles from VBOs
        gl.glEnable(GL2.GL_BLEND);
        gl.glBlendFunc(GL2.GL_SRC_ALPHA, GL2.GL_ONE_MINUS_SRC_ALPHA);
        gl.glEnable(GL2.GL_POINT_SMOOTH);
        gl.glPointSize(pointSize);
        scene.interact(gl);
        // 
        gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, getPosGLind());
        gl.glVertexPointer(4, GL2.GL_FLOAT, 0, 0);
        // 
        gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, getColGLind());
        gl.glColorPointer(4, GL2.GL_FLOAT, 0, 0);
        // Redraw
        if(posBuffSize > 0){
            gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
        }
        if(colBuffSize > 0){
            gl.glEnableClientState(GL2.GL_COLOR_ARRAY);
        }
        gl.glDisableClientState(GL2.GL_NORMAL_ARRAY);
        
        gl.glDrawArrays(GL2.GL_POINTS, 0, posBuffSize);
        
        if(posBuffSize > 0){
            gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);
        }if(colBuffSize > 0){
            gl.glDisableClientState(GL2.GL_COLOR_ARRAY);
        }
    }
    private void pushPerspectiveView(GL2 gl) {
        gl.glMatrixMode(GL2.GL_PROJECTION);
        gl.glPushMatrix();
        gl.glLoadIdentity();
        glu.gluPerspective(60, width / (float)height, 1, 1000);
        gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glPushMatrix();
        gl.glLoadIdentity();
    }
    private void popView(GL2 gl) {
        gl.glMatrixMode(GL2.GL_PROJECTION);
        gl.glPopMatrix();
        gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glPopMatrix();
    }
    /**
    *         setSwapInterval sets the swap interval for opengl.
    * 
    *  @param sw - new swap interval time
    */ 
    public void setSwapInterval(int sw) {
        swapInterval = sw;
        updateSwapInt.set(true);
    }
    /**
    *         setPointSize sets the size of points drawn.
    * 
    *  @param pointS - size of points to be drawn
    */ 
    public void setPointSize(float pointS) {
        pointSize = pointS;
    }
    /**
    *         reshape is the callback for resizing of OpenGL drawable.
    * 
    *  @param drawable - drawable
    */ 
    @Override
    public void reshape(GLAutoDrawable drawable, int arg1, int arg2, int width, int height) {
        this.width = width;
        this.height = height;
        GL2 gl = drawable.getGL().getGL2();
        popView(gl);
        pushPerspectiveView(gl);
    }
    /**
    *         reshape is the callback for disposing of GL environment.
    * 
    *  @param drawable - drawable
    */ 
    @Override
    public void dispose(GLAutoDrawable drawable){}
}
