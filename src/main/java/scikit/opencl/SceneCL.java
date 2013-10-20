package scikit.opencl;

/**
* @(#)  SceneCL
*/

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.swing.JComponent;
import javax.swing.JPopupMenu;
import javax.swing.Timer;
import scikit.graphics.Scene;
import scikit.util.Bounds;

/**
*       SceneCL is an OpenCL extension of the scene class .
* 
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2013-07
*/
abstract public class SceneCL<T> extends Scene{
        private String _title;
        protected String initDrawKernel;
        protected int initDrawKernelPosBufferNum;
        protected int initDrawKernelColBufferNum;
        protected CLHelper clhelper = null;
        protected CLScheduler clscheduler = null;
        protected JComponent _component; // contains scene and possible other GUI objects
	protected JComponent _canvas;    // the canvas on which scene is drawn
	protected JPopupMenu _popup = new JPopupMenu();
	protected BasicEventListenerCLGL _bel;
        
        protected Timer _animateTimer = new Timer(50, new ActionListener() {
		public void actionPerformed(ActionEvent e) {
			animate();
			System.out.println("timeout!");
		}
	});
        
        public SceneCL(String title, CLHelper clin){
            super(title);
            clhelper = clin;
	}
                
        /**
        *         initCLGL does all the initializing of OpenCL/OpenGL necessary for the 
        *   creation of the scene. Needs to do initialization of OpenCL/OpenGL CLGL context 
        *   needs to occur here. Done using CLHelper.
        * 
        *   @param drawable - input drawable from GLEventListener
        */
        abstract public void initCLGL(GLAutoDrawable drawable);

        @Override
         /**
	 * Returns the portion of the scene volume which is currently being viewed.
	 * @return the view bounds
	 */
	abstract public Bounds viewBounds();
	
	/**
	 * Creates the canvas GUI component on which the scene will be drawn.  This object
	 * may display a pop-up menu when requested.
	 * @return the canvas GUI component
	 */
	abstract protected JComponent createCanvas();
	        
        /**
        *         interact is an interface method with UI like the BasicUSI and OpenGL. 
        * 
        *   @param gl - GL2 obj
        */
        abstract protected void interact(GL2 gl);
        
        /**
        *         getCLHelper returns the clhelper object managing this scene which has the proper context.
        * 
        *   @return clhelper object managing this scene which has the proper context.
        */
        public CLHelper getCLHelper(){
            return clhelper;
        }
        /**
        *         getEventListener returns the GLEventListener managing this scene which has the proper context.
        * 
        *   @return BasicEvenListener object managing this scene which has the proper context.
        */
        public BasicEventListenerCLGL getEventListener(){
            return _bel;
        }
        /**
        *         getCLScheduler returns the clscheduler object that can schedule event for this scene 
        *   which has the proper context and timing.
        * 
        *   @return clscheduler object managing this scene which has the proper context and timing.
        */
        public CLScheduler getCLScheduler(){
            return clscheduler;
        }
        /**
        *         getInitDrawKernelName returns the kernel which initializes the drawing of the scene.
        * 
        *   @return string with name of the kernel which initializes the drawing of the scene
        */
        public String getInitDrawKernelName(){
            return initDrawKernel;
        }        
        /**
        *         getInitDrawPosBuffNum returns the float buffer number for the position buffer 
        *   of the kernel which initializes the drawing of the scene.
        * 
        *   @return float buffer number for the position buffer of the kernel which initializes the drawing of the scene
        */
        public int getInitDrawPosBuffNum(){
            return initDrawKernelPosBufferNum;
        }
       /**
        *         getInitDrawColBuffNum returns the float buffer number for the color buffer 
        *   of the kernel which initializes the drawing of the scene.
        * 
        *   @return float buffer number for the color buffer of the kernel which initializes the drawing of the scene
        */
        public int getInitDrawColBuffNum(){
            return initDrawKernelColBufferNum;
        }
}