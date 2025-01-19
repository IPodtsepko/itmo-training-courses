merge into Marks
using NewMarks
on
    Marks.StudentId = NewMarks.StudentId
    and Marks.CourseId = NewMarks.CourseId
when matched and Marks.Mark < NewMarks.Mark then
update
    set Mark = NewMarks.Mark
when not matched then
insert
    (StudentId, CourseId, Mark)
values
    (
        NewMarks.StudentId,
        NewMarks.CourseId,
        NewMarks.Mark
    );
