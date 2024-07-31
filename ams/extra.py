from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
import pandas as pd
from app.helpers import convert_warehouse_layout_to_matrix,convert_warehouse_layout_to_matrix_0_1, find_item_coordinates, get_cell_value, merge_excel_files, search_text_in_column
import os
from django.conf import settings
from django.contrib import messages

from app.algo import MazeSolver, horizontal_closest_point, nearest_neighbor, update_maze_with_checkpoints, vertical_closest_point
from app.asile import reduce_variables
from app.downloadfile import color_coordinates, color_coordinates_heatmap
from app.distance import extract_subpaths, print_path_with_distances




# Home Page
def home(request):
    # GET METHOD ONLY   
    picklist_name = request.session.get('picklist-name')
    layout_name = request.session.get('layout-name')
    context = {
        'layout_name':layout_name,
        'picklist_name':picklist_name,
    }
    return render(request,'home.html',context)





# For showing uploaded layout preview
def layout(request):
    #  IN THIS GET GRID LAYOUT METHOD WILL WORK
    return render(request,'layoutPage.html')



#  For uploading Picklist and showing it
def picklist(request):
    # POST METHOD
    if request.method == 'POST':
        uploaded_file = request.FILES.get('picklistFile')

        if uploaded_file:
            file_path = os.path.join(settings.MEDIA_ROOT, 'picklist', uploaded_file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            

            df = pd.read_excel(uploaded_file)

            # Convert DataFrame to HTML
            picklist_data = df.to_html(index=False)
            # picklist_data = fetch_column_from_excel(file_path)


            if picklist_data:
                
                # storing picklist data and its name in session
                request.session['picklist-name'] = uploaded_file.name
                request.session['picklist-data'] = picklist_data
                request.session['picklist-path'] = file_path
                picklist_data = request.session.get('picklist-data')
                # print(picklist_data)
                messages.success(request, 'Picklist uploaded successfully!')
                return redirect('picklist')
            else:   
                messages.error(request,'Picklist Not Uploaded Sucessfully')
                
                return redirect('picklist')
    # GET METHOD
    else:  
        picklist_data = request.session.get('picklist-data')
        picklist_name = request.session.get('picklist-name')
        context = {
            'picklist_data':picklist_data,
            'picklist_name':picklist_name,
        }
        return render(request,'picklistPage.html',context)





# For downloading the file that contains optimized path
def download_layout(request):
    # Path to the layout Excel file
    layout_file_path = request.session.get('layout-path')
    checkpoints = request.session.get('checkpoints')
    solution_path = request.session.get('solution-path')
    
    path = color_coordinates(layout_file_path , checkpoints, solution_path)
    with open(path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        # Specify the filename for the downloaded file
        response['Content-Disposition'] = 'attachment; filename="OptimizedPath.xlsx"'
        return response
    

# For downloading the file that contains optimized path
def download_distance_layout(request):
    # Path to the layout Excel file
    
    distance_path = request.session.get('distance_path') 
    picklist_path = request.session.get('picklist-path') 
    # print(distance_path, picklist_path)
    result_file = merge_excel_files(distance_path,picklist_path)
    # print(distance_path)
    
    # Open the layout file in binary mode
    with open(result_file, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        # Specify the filename for the downloaded file
        response['Content-Disposition'] = 'attachment; filename="merged_file.xlsx"'
        return response


# For downloading the file that contains optimized path
def download_heatmap(request):

    heatmap = request.session.get('heatmap')
    layout_file_path = request.session.get('layout-path')
    path = color_coordinates_heatmap(layout_file_path ,heatmap)
    
    with open(path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename="heatmap.xlsx"'
        return response


#  For uploading the warehouse layout
def upload_layout(request):
    # POST METHOD
    if request.method == 'POST':
        layout_file = request.FILES.get('layoutFile')
        unit_width = request.POST.get('unit-width')
        frame_depth = request.POST.get('frame-depth')
        col_asile = request.POST.get('col-asile')
        row_asile = request.POST.get('row-asile')
        image_selection = request.POST.get('imageSelection')


        if layout_file:
            file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', layout_file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in layout_file.chunks():
                    destination.write(chunk)

            warehouse_matrix = convert_warehouse_layout_to_matrix(file_path)
            
            # storing warehouse matrix and layout name in session
            request.session['warehouse_matrix'] = warehouse_matrix
            request.session['layout-name'] = layout_file.name
            request.session['layout-path'] = file_path
            request.session['unit_width'] = unit_width
            request.session['frame_depth'] = frame_depth
            request.session['row_asile'] = row_asile
            request.session['col_asile'] = col_asile
            request.session['image_selection'] = image_selection

        return redirect('layout')
    # GET METHOD
    else:
        return redirect('home')




# Admin page for Uploading Warehouse
def warehouse_admin(request):
    layout_name = request.session.get('layout-name')
    context = {
        'layout_name':layout_name
    }
    return render(request,'adminPage.html',context)




# Warehouse Optimized path 
def warehouse_map(request):
    try:

        layout_path = request.session.get('layout-path')
        picklist_path = request.session.get('picklist-path')

        # 0 and 1 Layout
        maze = convert_warehouse_layout_to_matrix_0_1(layout_path)

        unit_width = request.session.get('unit_width')
        frame_depth = request.session.get('frame_depth')
        col_asile = request.session.get('col_asile')
        row_asile = request.session.get('row_asile')
        image_selection = request.session.get('image_selection')

        unit_width = float(unit_width)
        frame_depth = float(frame_depth)
        col_asile = float(col_asile)
        row_asile = float(row_asile)


        output_coordinates, start_coordinates, goal_coordinates = find_item_coordinates(input_file_path_1=picklist_path, input_file_path_2=layout_path)
        

        # START WORKING ON ALGORITHM

        checkpoints = nearest_neighbor(start_coordinates, output_coordinates)
        request.session['checkpoints'] = checkpoints
        # print("Checkpoint = ",checkpoints)

        if image_selection == "vertical":
            checkpoints, direction_arr = horizontal_closest_point(maze, checkpoints)
            # print(checkpoints, '\n', direction_arr, 'leftright')
        elif image_selection == "horizontal":
            checkpoints, direction_arr = vertical_closest_point(maze, checkpoints)
            # print(checkpoints, '\n', direction_arr, 'updown')


        # print("Checkpoint after = ",checkpoints)
        
        ans = update_maze_with_checkpoints(maze, checkpoints, start_coordinates, goal_coordinates)
        # print('ans = ',ans)
        
        request.session['start'] = start_coordinates
        request.session['goal'] = goal_coordinates


        # print(maze)
        horizontal_weight, vertical_weight = reduce_variables(maze,column_aisle_width=col_asile,row_aisle_width=row_asile,frame_depth=frame_depth,beam_width=unit_width)
        
        solver = MazeSolver(maze,horizontal_weight=horizontal_weight,vertical_weight=vertical_weight)
        # print(maze)

        found_path, solution_path, total_steps, total_distance = solver.solve_maze_no_visualization(
            start_coordinates, goal_coordinates, checkpoints
        )
        # print(start_coordinates)
        # print(goal_coordinates)
        # print("S " ,solution_path)
        # print("C = ",checkpoints)
        subpaths = extract_subpaths(solution_path, checkpoints, start_coordinates, goal_coordinates, vertical_weight, horizontal_weight)
        
        # print(subpaths)

        request.session['solution-path'] = solution_path

        if found_path:
            # print(solution_path)
            request.session['path'] = solution_path
            # print('path found')
        else:
            print("No path found.")

        coordinates = request.session.get('checkpoints')
        # print(coordinates)
        bin_names = get_cell_value(layout_path, coordinates)

        output, distance_path = print_path_with_distances("START","GOAL", bin_names, subpaths, row_asile)
        request.session['distance_path'] = distance_path
        # print(distance_path)
        total_distance = sum(subpaths)
        
        context = {
            'checkpoints_info':output,
            'total_distance':total_distance,
        }
    
    except Exception as e:
        print(e)
        return render (request,'other.html')
    
    return render(request,'warehouse-map.html',context)



def error(request):
    return render(request,'error.html')


#  Fetch fn using js for creating UI of uploaded layout
def get_grid_data(request):
    warehouse_matrix = request.session.get('warehouse_matrix')
    if warehouse_matrix:
        grid_data = warehouse_matrix
    else:
        grid_data = None

    layout_type = request.session.get('image_selection')
    response_data = {
        "layout_type":layout_type,
        "gridData":grid_data
    }
        
    return JsonResponse(response_data, safe=False)


#  Checkpoints and path and start and end using fetch js fn
def get_coordinates(request):
    path = request.session.get('path')
    checkpoints = request.session.get('checkpoints')
    # picklist_path = request.session.get('picklist-path')
    
    # checkpoint_with_desc = search_text_in_column(picklist_path)
    # print("checkpont " , checkpoint_with_desc)
    # request.session['checkpoint_with_desc'] = checkpoint_with_desc

    start = request.session.get('start')
    goal = request.session.get('goal')
    points = [
        start,goal
    ]
    response_data = {
        "coordinates": path,
        "points": points,
        "checkpoints":checkpoints,
        # "checkpoint_with_desc":checkpoint_with_desc,
    }
    return JsonResponse(response_data, safe=False)




def heatmap(request):
    # POST METHOD
    if request.method == 'POST':
        uploaded_file = request.FILES.get('heatmapfile')

        if uploaded_file:
            file_path = os.path.join(settings.MEDIA_ROOT, 'heatmap', uploaded_file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
                request.session['heatmap-name'] = uploaded_file.name
                request.session['heatmap-path'] = file_path
                messages.success(request, 'heatmap uploaded successfully!')
                
        else:   
            messages.error(request,'heatmap Not Uploaded Sucessfully')
                
        return redirect('heatmap')
    # GET METHOD
    else:  
        heatmap_name = request.session.get('heatmap-name')
        context = {
            'heatmap_name':heatmap_name,
        }
        return render(request,'heatmap.html',context)




def show_heatmap(request):
    return render(request,'show_heatmap.html')


def getHeatmap(request):
    warehouse_matrix = request.session.get('warehouse_matrix')
    if warehouse_matrix:
        grid_data = warehouse_matrix
    else:
        grid_data = None


    heatmap_path = request.session.get('heatmap-path')
    
    heatmap = search_text_in_column(heatmap_path)
    # print("checkpont " , checkpoint_with_desc)
    request.session['heatmap'] = heatmap

    layout_type = request.session.get('image_selection')
    response_data = {
        "layout_type":layout_type,
        "gridData":grid_data,
        "colorGrid":heatmap,
    }
        
    return JsonResponse(response_data, safe=False)